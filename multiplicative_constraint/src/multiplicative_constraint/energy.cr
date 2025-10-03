require "set"

module MultiplicativeConstraint
  struct Evaluation
    getter segments : Array(Array(Int32))
    getter labels : Array(Int32)
    getter spectral : Float64
    getter fairness : Float64
    getter weight_fairness : Float64
    getter entropy : Float64
    getter penalty : Float64
    getter cross_conflict : Float64
    getter unified : Float64

    def initialize(@segments, @labels, @spectral, @fairness, @weight_fairness, @entropy, @penalty, @cross_conflict, @unified)
    end
  end

  class Energy
    getter graph : Graph
    getter segments : Int32

    @size : Int32

    def initialize(@graph : Graph, @segments : Int32)
      @size = @graph.size
    end

    def unified(alpha : Array(Float64))
      evaluate(alpha).unified
    end

    def spectral(alpha : Array(Float64))
      evaluate(alpha).spectral
    end

    def count_fairness(alpha : Array(Float64))
      evaluate(alpha).fairness
    end

    def weight_fairness(alpha : Array(Float64))
      evaluate(alpha).weight_fairness
    end

    def entropy(alpha : Array(Float64))
      evaluate(alpha).entropy
    end

    def penalty(alpha : Array(Float64))
      evaluate(alpha).penalty
    end

    def cross_conflict(alpha : Array(Float64))
      evaluate(alpha).cross_conflict
    end

    def segments(alpha : Array(Float64))
      evaluate(alpha).segments
    end

    def evaluate(alpha : Array(Float64))
      segs = cuts_from_alpha(alpha)
      labels = Array.new(@size, 0)
      segs.each_with_index do |segment, idx|
        segment.each { |i| labels[i] = idx }
      end

      size_target = @size.to_f64 / @segments
      weight_target = @graph.weights.sum / @segments

      segment_sizes = Array(Float64).new(segs.size) { 0.0 }
      segment_weights = Array(Float64).new(segs.size) { 0.0 }

      segs.each_with_index do |segment, idx|
        segment_sizes[idx] = segment.size.to_f64
        weight = segment.sum { |i| @graph.weights[i] }
        segment_weights[idx] = weight
      end

      fairness = segment_sizes.reduce(0.0) do |sum, s|
        diff = s - size_target
        sum + diff * diff
      end / 2.0

      weight_fairness = segment_weights.reduce(0.0) do |sum, w|
        diff = w - weight_target
        sum + diff * diff
      end / 2.0

      total_size = segment_sizes.sum
      entropy = if total_size <= 0
                  0.0
                else
                  segment_sizes.reduce(0.0) do |sum, size|
                    next sum if size <= 0
                    p = size / total_size
                    sum - p * Math.log(p)
                  end
                end

      penalty_product = segs.reduce(1.0) do |product, segment|
        next product if segment.empty?
        factor = segment.reduce(1.0) do |value, idx|
          weight = @graph.weights[idx]
          value * (1.0 - 1.0 / (weight * weight))
        end
        product * factor
      end

      degrees = Array.new(@size, 0.0)
      cross_conflict = 0.0
      @graph.edges.each do |edge|
        i, j, weight = edge
        if labels[i] == labels[j]
          degrees[i] += weight
          degrees[j] += weight
        else
          cross_conflict += weight
        end
      end

      spectral = -heat_trace(labels, degrees, samples: 4, order: 6)
      unified = spectral + fairness + 0.5 * weight_fairness - 0.1 * entropy - penalty_product

      Evaluation.new(
        segments: segs,
        labels: labels,
        spectral: spectral,
        fairness: fairness,
        weight_fairness: weight_fairness,
        entropy: entropy,
        penalty: penalty_product,
        cross_conflict: cross_conflict,
        unified: unified
      )
    end

    private def heat_trace(labels, degrees, samples = 4, order = 6)
      samples = Math.max(1, samples)
      seed = labels.reduce(17_u64) { |acc, val| (acc &* 31_u64) ^ val.to_u64 }
      random = Random.new(seed)
      factorial = 1.0
      sample_sum = 0.0
      samples.times do
        vector = Array(Float64).new(@size) { random.rand < 0.5 ? -1.0 : 1.0 }
        current = vector.dup
        factorial = 1.0
        accum = 0.0
        (0..order).each do |k|
          coefficient = k.even? ? 1.0 : -1.0
          accum += coefficient / factorial * dot(vector, current)
          break if k == order
          current = masked_laplacian_apply(labels, degrees, current)
          factorial *= (k + 1).to_f
        end
        sample_sum += accum
      end
      sample_sum / samples
    end

    private def segments_from_cuts(cuts)
      return [Array(Int32).new] if cuts.empty?
      segments = Array(Array(Int32)).new
      cuts.each_with_index do |start, idx|
        endpoint = cuts[(idx + 1) % cuts.size]
        segment = if start == endpoint
                    Array(Int32).new
                  elsif start < endpoint
                    (start...endpoint).to_a
                  else
                    ((start...@size).to_a + (0...endpoint).to_a)
                  end
        segments << segment
      end
      segments
    end

    private def cuts_from_alpha(alpha)
      normalized = alpha.map { |a| (a % (2 * Math::PI)).to_f64 }
      scaled = normalized.map do |norm|
        value = ((norm / (2 * Math::PI)) * @size).floor
        value = value.clamp(0.0, (@size - 1).to_f64)
        value.to_i
      end

      adjusted = adjust_indices(scaled)
      adjusted = [0] if adjusted.empty?
      segments_from_cuts(adjusted.sort)
    end

    private def masked_laplacian_apply(labels, degrees, vector)
      result = Array.new(@size) { |i| degrees[i] * vector[i] }
      @graph.edges.each do |edge|
        i, j, weight = edge
        next unless labels[i] == labels[j]
        result[i] -= weight * vector[j]
        result[j] -= weight * vector[i]
      end
      result
    end

    private def dot(a, b)
      sum = 0.0
      @size.times { |i| sum += a[i] * b[i] }
      sum
    end

    private def adjust_indices(indices : Array(Int32))
      return [] of Int32 if indices.empty? || @size <= 0

      used = Set(Int32).new
      adjusted = Array(Int32).new(indices.size)

      indices.sort.each do |value|
        candidate = ((value % @size) + @size) % @size
        if @size > 0
          while used.includes?(candidate) && used.size < @size
            candidate = (candidate + 1) % @size
          end
        end
        used.add(candidate)
        adjusted << candidate
        break if used.size == @size
      end

      adjusted
    end
  end
end
