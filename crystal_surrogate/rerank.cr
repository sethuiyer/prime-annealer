#!/usr/bin/env crystal

require "json"
require "../multiplicative_constraint/src/multiplicative_constraint"

module CrystalSurrogate
  include MultiplicativeConstraint

  SEGMENTS = 3
  CANDIDATES = 200
  TOP_K = 10

  class MLP
    property w1 : Array(Array(Float64))
    property b1 : Array(Float64)
    property w2 : Array(Float64)
    property b2 : Float64

    def initialize(input_dim : Int32, hidden : Int32)
      @w1 = Array.new(hidden) { Array.new(input_dim, 0.0) }
      @b1 = Array.new(hidden, 0.0)
      @w2 = Array.new(hidden, 0.0)
      @b2 = 0.0
    end

    def self.load(path : String)
      content = File.read(path)
      json = JSON.parse(content).as_h
      hidden = json["w1"].as_a.size
      input_dim = json["w1"].as_a.first.as_a.size
      model = self.new(input_dim, hidden)
      model.w1 = json["w1"].as_a.map(&.as_a.map(&.as_f))
      model.b1 = json["b1"].as_a.map(&.as_f)
      model.w2 = json["w2"].as_a.map(&.as_f)
      model.b2 = json["b2"].as_f
      model
    end

    def forward(x : Array(Float64))
      z1 = Array(Float64).new(@w1.size) do |i|
        dot = 0.0
        row = @w1[i]
        x.each_with_index { |v, j| dot += row[j] * v }
        dot + @b1[i]
      end
      h = z1.map { |v| v > 0 ? v : 0.0 }
      out = @w2.each_with_index.reduce(@b2) do |sum, (w, k)|
        sum + w * h[k]
      end
      out
    end

    def predict(x : Array(Float64)) : Float64
      forward(x)
    end
  end

  def self.features_from_alpha(alpha : Array(Float64))
    alpha.flat_map { |a| [Math.cos(a), Math.sin(a)] }
  end

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

  def self.run
    model_path = ENV["SURROGATE_MODEL"]? || raise "set SURROGATE_MODEL to saved_state.json"
    model = MLP.load(model_path)

    graph = build_graph(24)
    energy = Energy.new(graph, SEGMENTS)

    rng = Random::PCG32.new

    candidates = Array.new(CANDIDATES) do
      alpha = Array.new(SEGMENTS) { rng.rand * 2.0 * Math::PI }
      features = features_from_alpha(alpha)
      pred = model.predict(features)
      {alpha: alpha, features: features, pred: pred}
    end

    candidates.sort_by! { |cand| cand[:pred] }

    puts "Top #{TOP_K} candidates by surrogate prediction:" 
    candidates.first(TOP_K).each_with_index do |cand, idx|
      actual = energy.unified(cand[:alpha])
      puts "  #{idx + 1}. predicted=#{cand[:pred]}, actual=#{actual}" 
    end
  end
end

CrystalSurrogate.run
