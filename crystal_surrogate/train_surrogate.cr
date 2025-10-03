#!/usr/bin/env crystal

require "json"
require "../multiplicative_constraint/src/multiplicative_constraint"

module CrystalSurrogate
  include MultiplicativeConstraint

  SAMPLE_COUNT   = 1600
  TRAIN_RATIO    = 0.8
  SEGMENTS       = 3
  HIDDEN_UNITS   = 24
  EPOCHS         = 200
  BATCH_SIZE     = 32
  LEARNING_RATE  = 5e-3

  # Simple utility to build a necklace-style graph with the first `count` primes.
  def self.build_graph(count : Int32)
    weights = Weights.assign(count)

    adjacency = Array.new(count) { Array.new(count, 0.0_f64) }
    gaps = Array(Float64).new(count, 0.0)
    (0...(count - 1)).each do |i|
      gaps[i] = (weights[i + 1] - weights[i]).abs
    end
    gaps[count - 1] = gaps[count - 2]

    count.times do |i|
      j = (i + 1) % count
      adjacency[i][j] = adjacency[j][i] = 1.0 + gaps[i]
    end

    count.times do |i|
      j = (i + 2) % count
      avg_gap = (gaps[i] + gaps[(i + 1) % count]) / 2.0
      adjacency[i][j] = adjacency[j][i] = 0.5 * (adjacency[i][(i + 1) % count] + (1.0 + avg_gap))
    end

    Graph.new(weights, adjacency)
  end

  # Feed-forward two-layer network with manual SGD.
  class MLP
    property w1 : Array(Array(Float64))
    property b1 : Array(Float64)
    property w2 : Array(Float64)
    property b2 : Float64

    def initialize(input_dim : Int32, hidden : Int32)
      rng = Random::PCG32.new
      scale1 = Math.sqrt(2.0 / input_dim)
      scale2 = Math.sqrt(2.0 / hidden)

      @w1 = Array.new(hidden) do
        Array.new(input_dim) { (rng.rand - 0.5) * scale1 }
      end
      @b1 = Array.new(hidden, 0.0)
      @w2 = Array.new(hidden) { (rng.rand - 0.5) * scale2 }
      @b2 = 0.0
    end

    def forward(x : Array(Float64))
      z1 = Array(Float64).new(@w1.size) do |i|
        dot = 0.0
        row = @w1[i]
        x.each_with_index { |v, j| dot += row[j] * v }
        dot + @b1[i]
      end

      h = z1.map { |v| v > 0 ? v : 0.0 }

      out = @w2.each_with_index.reduce(@b2) do |sum, (w, i)|
        sum + w * h[i]
      end

      {z1: z1, hidden: h, output: out}
    end

    def train(features : Array(Array(Float64)), targets : Array(Float64))
      total = features.size
      batch_count = Math.max(total // BATCH_SIZE, 1)

      EPOCHS.times do |epoch|
        sum_loss = 0.0
        order = (0...total).to_a.shuffle

        batch_count.times do |batch_idx|
          start = batch_idx * BATCH_SIZE
          finish = Math.min(start + BATCH_SIZE, total)

          grad_w1 = Array.new(@w1.size) { Array.new(@w1.first.size, 0.0) }
          grad_b1 = Array.new(@b1.size, 0.0)
          grad_w2 = Array.new(@w2.size, 0.0)
          grad_b2 = 0.0

          (start...finish).each do |pos|
            i = order[pos]
            x = features[i]
            y_true = targets[i]
            cache = forward(x)
            y_pred = cache[:output]
            diff = y_pred - y_true
            sum_loss += 0.5 * diff * diff

            cache_hidden = cache[:hidden]
            cache_z1 = cache[:z1]

            cache_hidden.each_with_index do |h_val, idx|
              grad_w2[idx] += diff * h_val
            end
            grad_b2 += diff

            cache_hidden.each_with_index do |_, idx|
              dh = diff * @w2[idx]
              relu_grad = cache_z1[idx] > 0 ? 1.0 : 0.0
              delta = dh * relu_grad
              grad_b1[idx] += delta
              x.each_with_index do |xj, j|
                grad_w1[idx][j] += delta * xj
              end
            end
          end

          scale = LEARNING_RATE / (finish - start)

          @w2.each_index do |i|
            @w2[i] -= scale * grad_w2[i]
          end
          @b2 -= scale * grad_b2

          @w1.each_index do |i|
            row = @w1[i]
            grad_row = grad_w1[i]
            row.each_index do |j|
              row[j] -= scale * grad_row[j]
            end
            @b1[i] -= scale * grad_b1[i]
          end
        end

        avg_loss = sum_loss / total
        puts "Epoch #{epoch + 1}/#{EPOCHS} - avg_loss=#{avg_loss.round(6)}" if (epoch % 20).zero?
      end
    end

    def predict(x : Array(Float64)) : Float64
      forward(x)[:output]
    end

    def export_state
      JSON.build do |json|
        json.object do
          json.field("w1") do
            json.array do
              @w1.each do |row|
                json.array do
                  row.each { |val| json.number(val) }
                end
              end
            end
          end
          json.field("b1") do
            json.array { @b1.each { |val| json.number(val) } }
          end
          json.field("w2") do
            json.array { @w2.each { |val| json.number(val) } }
          end
          json.field("b2") { json.number(@b2) }
        end
      end
    end

    def self.load_state(state : Hash(String, JSON::Any), input_dim : Int32, hidden : Int32)
      model = self.new(input_dim, hidden)
      model.w1 = state["w1"].as_a.map(&.as_a.map(&.as_f64))
      model.b1 = state["b1"].as_a.map(&.as_f64)
      model.w2 = state["w2"].as_a.map(&.as_f64)
      model.b2 = state["b2"].as_f64
      model
    end
  end

  def self.features_from_alpha(alpha : Array(Float64))
    alpha.flat_map { |a| [Math.cos(a), Math.sin(a)] }
  end

  def self.generate_dataset(energy : Energy, count : Int32)
    rng = Random::PCG32.new
    samples = Array(Array(Float64)).new(count)
    targets = Array(Float64).new(count)

    count.times do
      alpha = Array.new(SEGMENTS) { rng.rand * 2.0 * Math::PI }
      samples << features_from_alpha(alpha)
      targets << energy.unified(alpha)
    end

    {samples: samples, targets: targets}
  end

  def self.split_dataset(samples, targets)
    total = samples.size
    train_size = (total * TRAIN_RATIO).to_i
    train_features = samples[0, train_size]
    train_targets = targets[0, train_size]
    val_features = samples[train_size, total - train_size]
    val_targets = targets[train_size, total - train_size]
    {train_features: train_features, train_targets: train_targets,
     val_features: val_features, val_targets: val_targets}
  end

  def self.mean(arr : Array(Float64))
    arr.sum / arr.size
  end

  def self.stddev(arr : Array(Float64), mean_val : Float64)
    Math.sqrt(arr.reduce(0.0) { |acc, v| acc + (v - mean_val) ** 2 } / arr.size)
  end

  def self.correlation(a : Array(Float64), b : Array(Float64))
    raise "size mismatch" unless a.size == b.size
    mean_a = mean(a)
    mean_b = mean(b)
    std_a = stddev(a, mean_a)
    std_b = stddev(b, mean_b)
    cov = a.zip(b).reduce(0.0) { |acc, (x, y)| acc + (x - mean_a) * (y - mean_b) } / a.size
    cov / (std_a * std_b)
  end

  def self.run
    save_path = ENV["SURROGATE_EXPORT_PATH"]?
    graph = build_graph(24)
    energy = Energy.new(graph, SEGMENTS)

    dataset = generate_dataset(energy, SAMPLE_COUNT)
    split = split_dataset(dataset[:samples], dataset[:targets])

    mean_target = mean(split[:train_targets])
    std_target = stddev(split[:train_targets], mean_target)
    std_target = 1.0 if std_target.abs < 1e-6

    normalized_train_targets = split[:train_targets].map { |t| (t - mean_target) / std_target }

    model = MLP.new(split[:train_features].first.size, HIDDEN_UNITS)
    puts "Training surrogate on #{split[:train_features].size} samples"
    model.train(split[:train_features], normalized_train_targets)

    normalized_predictions = split[:val_features].map { |feat| model.predict(feat) }
    predictions = normalized_predictions.map { |p| p * std_target + mean_target }
    corr = correlation(predictions, split[:val_targets])

    puts "Validation correlation (pred vs true energy): #{corr.round(4)}"

    split[:val_features].zip(split[:val_targets], predictions).first(5).each_with_index do |(_, target, pred), idx|
      puts "Sample #{idx + 1}: target=#{target.round(4)}, predicted=#{pred.round(4)}"
    end

    if save_path
      state = model.export_state
      File.write(save_path, state)
      puts "Saved surrogate parameters to #{save_path}"
    end
  end
end

CrystalSurrogate.run
