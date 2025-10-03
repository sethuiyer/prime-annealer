# Heat-kernel spectral partitioning example for ASIC floorplanning
# Implemented in Crystal

struct NetSpec
  getter name : String
  getter gate_count : Int32
  getter toggle_rate : Float64
  getter regions : Array(String)

  def initialize(@name, @gate_count, @toggle_rate, regions : Array(String))
    @regions = regions
  end
end

module DeterministicWeights
  extend self

  def primes(limit : Int32) : Array(Int32)
    sieve = Array.new(limit + 1, true)
    sieve[0] = sieve[1] = false
    2.upto(Math.sqrt(limit).to_i) do |p|
      next unless sieve[p]
      (p * p).step(to: limit, by: p) { |k| sieve[k] = false }
    end
    primes = Array(Int32).new
    sieve.each_with_index do |flag, idx|
      primes << idx if flag
    end
    primes
  end

  def assign(count : Int32) : Array(Float64)
    # generate more than needed to cover count
    pool = primes(20 * count)
    raise "not enough weight seeds" if pool.size < count
    pool.first(count).map(&.to_f64)
  end
end

module LinearAlgebra
  extend self

  def identity(n : Int32) : Array(Array(Float64))
    Array.new(n) do |i|
      Array.new(n) { |j| i == j ? 1.0 : 0.0 }
    end
  end

  def multiply(a : Array(Array(Float64)), b : Array(Array(Float64))) : Array(Array(Float64))
    n = a.size
    m = b.first.size
    k = b.size
    result = Array.new(n) { Array.new(m, 0.0) }
    n.times do |i|
      m.times do |j|
        sum = 0.0
        k.times do |t|
          sum += a[i][t] * b[t][j]
        end
        result[i][j] = sum
      end
    end
    result
  end

  def trace(matrix : Array(Array(Float64))) : Float64
    matrix.each_with_index.reduce(0.0) do |sum, (row, idx)|
      sum + row[idx]
    end
  end

  def copy(matrix : Array(Array(Float64))) : Array(Array(Float64))
    matrix.map(&.dup)
  end
end

def gaussian(rng : Random, scale : Float64) : Float64
  u1 = rng.rand.clamp(1e-9, 1.0)
  u2 = rng.rand
  magnitude = Math.sqrt(-2.0 * Math.log(u1))
  angle = 2.0 * Math::PI * u2
  magnitude * Math.cos(angle) * scale
end

class SpectralPartitioner
  getter nets : Array(NetSpec)
  getter weights : Array(Float64)
  getter conflict : Array(Array(Float64))

  def initialize(@nets)
    @weights = DeterministicWeights.assign(@nets.size)
    @conflict = build_conflict_matrix
  end

  def build_conflict_matrix : Array(Array(Float64))
    n = nets.size
    matrix = Array.new(n) { Array.new(n, 0.0) }
    n.times do |i|
      ((i + 1)...n).each do |j|
        overlap = (nets[i].regions & nets[j].regions).size
        next if overlap == 0
        # Weight conflicts by overlap and toggle-rate proximity
        toggle_factor = 1.0 + (nets[i].toggle_rate - nets[j].toggle_rate).abs
        weight = overlap.to_f64 * toggle_factor
        matrix[i][j] = matrix[j][i] = weight
      end
    end
    matrix
  end

  def laplacian(alpha : Array(Float64)) : Array(Array(Float64))
    labels = segment_labels(alpha)
    n = nets.size
    masked = Array.new(n) { Array.new(n, 0.0) }
    n.times do |i|
      n.times do |j|
        if labels[i] == labels[j]
          masked[i][j] = conflict[i][j]
        end
      end
    end

    degrees = Array.new(n, 0.0)
    n.times do |i|
      sum = 0.0
      n.times { |j| sum += masked[i][j] }
      degrees[i] = sum
    end

    lap = Array.new(n) { Array.new(n, 0.0) }
    n.times do |i|
      lap[i][i] = degrees[i]
      n.times do |j|
        next if i == j
        lap[i][j] = -masked[i][j]
      end
    end
    lap
  end

  def heat_trace(alpha : Array(Float64), order : Int32 = 6) : Float64
    lap = laplacian(alpha)
    n = lap.size
    current = LinearAlgebra.identity(n)
    trace_sum = 0.0
    factorial = 1.0
    # series for exp(-L) = sum (-1)^k L^k / k!
    (0..order).each do |k|
      trace_sum += ((k.even? ? 1.0 : -1.0) / factorial) * LinearAlgebra.trace(current)
      current = LinearAlgebra.multiply(current, lap) unless k == order
      factorial *= (k + 1).to_f unless k == order
    end
    trace_sum
  end

  def segment_labels(alpha : Array(Float64)) : Array(Int32)
    cuts = cuts_from_alpha(alpha)
    labels = Array.new(nets.size, 0)
    cuts.each_with_index do |segment, idx|
      segment.each { |index| labels[index] = idx }
    end
    labels
  end

  def cuts_from_alpha(alpha : Array(Float64)) : Array(Array(Int32))
    segments = alpha.size
    normalized = alpha.map { |a| (a % (2 * Math::PI)).to_f64 }
    scaled = normalized.map do |norm|
      value = ((norm / (2 * Math::PI)) * nets.size).floor
      value = value.clamp(0.0, (nets.size - 1).to_f64)
      value.to_i
    end
    unique = scaled.uniq.sort
    unique = (unique + [0]).uniq.sort if unique.empty?

    result = Array(Array(Int32)).new
    unique.each_with_index do |start, idx|
      endpoint = unique[(idx + 1) % unique.size]
      if start == endpoint
        result << Array(Int32).new
      elsif start < endpoint
        result << (start...endpoint).to_a
      else
        result << ((start...nets.size).to_a + (0...endpoint).to_a)
      end
    end
    result
  end

  def fairness_energy(alpha : Array(Float64)) : Float64
    sizes = cuts_from_alpha(alpha).map(&.size.to_f64)
    return 0.0 if sizes.empty?
    target = nets.size.to_f64 / alpha.size
    sizes.reduce(0.0) { |sum, s| diff = s - target; sum + diff * diff } / 2.0
  end

  def weight_fairness(alpha : Array(Float64)) : Float64
    segments = cuts_from_alpha(alpha)
    target = weights.sum / alpha.size
    segments.reduce(0.0) do |sum, seg|
      weight_sum = seg.sum { |idx| weights[idx] }
      diff = weight_sum - target
      sum + diff * diff
    end / 2.0
  end

  def entropy(alpha : Array(Float64)) : Float64
    sizes = cuts_from_alpha(alpha).map(&.size.to_f64)
    total = sizes.sum
    return 0.0 if total.zero?
    sizes.reduce(0.0) do |sum, size|
      p = size / total
      next sum if p <= 0
      sum - p * Math.log(p)
    end
  end

  def multiplicative_weights_penalty(alpha : Array(Float64)) : Float64
    cuts_from_alpha(alpha).reduce(1.0) do |product, seg|
      next product if seg.empty?
      segment_factor = seg.reduce(1.0) do |factor, idx|
        weight = weights[idx]
        factor * (1.0 - 1.0 / (weight * weight))
      end
      product * segment_factor
    end
  end

  def cross_conflict(alpha : Array(Float64)) : Float64
    labels = segment_labels(alpha)
    sum = 0.0
    nets.size.times do |i|
      ((i + 1)...nets.size).each do |j|
        sum += conflict[i][j] if labels[i] != labels[j]
      end
    end
    sum
  end

  def unified_energy(alpha : Array(Float64)) : Float64
    spectral = -heat_trace(alpha)
    balance = fairness_energy(alpha)
    weight_balance = 0.5 * weight_fairness(alpha)
    entropy_term = -0.1 * entropy(alpha)
    penalty_term = -multiplicative_weights_penalty(alpha)
    spectral + balance + weight_balance + entropy_term + penalty_term
  end

  def anneal(blocks : Int32, iterations : Int32 = 1500, step : Float64 = 0.35, seed : Int32 = 42) : Tuple(Array(Float64), Float64)
    rng = Random.new(seed)
    alpha = Array.new(blocks) { rng.rand * 2 * Math::PI }
    energy = unified_energy(alpha)
    best_alpha = alpha.dup
    best_energy = energy
    step_scale = step
    iterations.times do |iter|
      temperature = Math.max(0.02, 1.0 - iter / iterations.to_f64)
      candidate = alpha.map do |a|
        delta = gaussian(rng, step_scale * temperature)
        (a + delta) % (2 * Math::PI)
      end
      candidate_energy = unified_energy(candidate)
      if candidate_energy < energy || rng.rand < Math.exp(-(candidate_energy - energy) / temperature)
        alpha = candidate
        energy = candidate_energy
        if candidate_energy < best_energy
          best_energy = candidate_energy
          best_alpha = candidate.dup
        end
      end
      step_scale = Math.max(0.05, step_scale * 0.999)
    end
    {best_alpha, best_energy}
  end

  def report(alpha : Array(Float64))
    segments = cuts_from_alpha(alpha)
    puts "Segments:"
    segments.each_with_index do |seg, idx|
      next if seg.empty?
      total_gate = seg.sum { |i| nets[i].gate_count }
      avg_toggle = seg.sum { |i| nets[i].toggle_rate } / seg.size
      names = seg.map { |i| nets[i].name }
      puts "  Block #{idx + 1}: #{names.join(", ")}"
      puts "    gates=#{total_gate}, avg_toggle=#{avg_toggle.round(4)}, weight_score=#{seg.sum { |i| weights[i] }.round(2)}"
    end
    puts "  Cross-conflict weight: #{cross_conflict(alpha).round(3)}"
    puts "  Multiplicative penalty: #{multiplicative_weights_penalty(alpha).round(6)}"
    puts "  Unified energy: #{unified_energy(alpha).round(6)}"
  end
end

# --- Example netlist (non-trivial) ---
nets = [
  NetSpec.new("ALU_PIPE0", 3850, 0.42, ["core", "datapath"]),
  NetSpec.new("ALU_PIPE1", 3790, 0.47, ["core", "datapath"]),
  NetSpec.new("FPU_CTRL", 2100, 0.21, ["core", "fp_cluster"]),
  NetSpec.new("L1_MISS", 1450, 0.55, ["cache", "axi"]),
  NetSpec.new("L2_ARB", 2680, 0.37, ["cache", "noc"]),
  NetSpec.new("DMA_ENG", 1980, 0.63, ["noc", "periph"]),
  NetSpec.new("PCIe_RX", 2320, 0.18, ["periph", "serdes"]),
  NetSpec.new("PCIe_TX", 2290, 0.23, ["periph", "serdes"]),
  NetSpec.new("USB3_CORE", 1560, 0.41, ["periph", "serdes"]),
  NetSpec.new("DDR_PHY", 3010, 0.34, ["noc", "phy"]),
  NetSpec.new("DDR_CTRL", 4150, 0.52, ["noc", "phy"]),
  NetSpec.new("SEC_ACCEL", 2640, 0.49, ["periph", "sec"])
]

partitioner = SpectralPartitioner.new(nets)
blocks = 3
alpha, energy = partitioner.anneal(blocks, iterations: 2000, step: 0.4, seed: 2025)
puts "Best energy: #{energy.round(6)}"
partitioner.report(alpha)
