require "../src/multiplicative_constraint"

module LargePrimeNecklace
  include MultiplicativeConstraint

  MAX_VALUE = 10_000
  SEGMENTS = 24
  SEEDS = (4101..4106)

  def self.primes_below(limit : Int32) : Array(Int32)
    sieve = Array.new(limit + 1, true)
    sieve[0] = sieve[1] = false
    i = 2
    while i * i <= limit
      if sieve[i]
        k = i * i
        while k <= limit
          sieve[k] = false
          k += i
        end
      end
      i += 1
    end
    primes = Array(Int32).new
    sieve.each_with_index do |flag, idx|
      primes << idx if flag
    end
    primes
  end

  def self.build_weights(primes : Array(Int32)) : Array(Float64)
    primes.map(&.to_f64)
  end

  def self.build_adjacency(primes : Array(Int32)) : Array(Array(Float64))
    n = primes.size
    adjacency = Array.new(n) { Array.new(n, 0.0) }
    gaps = Array(Float64).new(n, 0.0)
    (0...n - 1).each do |i|
      gaps[i] = (primes[i + 1] - primes[i]).to_f64
    end
    gaps[n - 1] = gaps[n - 2]

    n.times do |i|
      j = (i + 1) % n
      w = 1.0 + gaps[i]
      adjacency[i][j] = adjacency[j][i] = w
    end

    n.times do |i|
      j = (i + 2) % n
      avg_gap = (gaps[i] + gaps[(i + 1) % n]) / 2.0
      weight = 0.5 * (adjacency[i][(i + 1) % n] + (1.0 + avg_gap))
      adjacency[i][j] = adjacency[j][i] = weight
    end

    adjacency
  end

  def self.labels(primes : Array(Int32)) : Array(String)
    primes.map_with_index { |p, idx| "p#{idx + 1}=#{p}" }
  end

  def self.run
    primes = primes_below(MAX_VALUE)
    weights = build_weights(primes)
    adjacency = build_adjacency(primes)
    names = labels(primes)

    runs = Array(NamedTuple(seed: Int32, energy: Float64, result: PartitionResult)).new
    SEEDS.each do |seed|
      graph = Graph.new(weights, adjacency)
      engine = Engine.new(graph, SEGMENTS)
      result = engine.solve(iterations: 1800, step: 0.28, seed: seed)
      runs << {seed: seed, energy: result.energy, result: result}
    end

    best = runs.min_by { |run| run[:energy] }
    unless best
      puts "No partitions generated"
      return
    end

    puts "Prime necklace (primes < #{MAX_VALUE}, count=#{primes.size}, segments=#{SEGMENTS})"
    runs.sort_by { |run| run[:energy] }.each do |run|
      puts "  seed=#{run[:seed]} -> energy=#{run[:energy]}"
    end
    puts
    result = best[:result]
    puts "Selected best configuration (seed=#{best[:seed]}):"
    puts "Unified energy: #{result.energy}"
    puts "Spectral action: #{result.spectral}"
    puts "Fairness energy: #{result.fairness}"
    puts "Weight fairness: #{result.weight_fairness}"
    puts "Entropy: #{result.entropy}"
    puts "Multiplicative penalty: #{result.penalty}"
    puts "Cross-conflict weight: #{result.cross_conflict}"
    puts
    puts "Segment summary:"
    result.segments.each_with_index do |segment, idx|
      next if segment.empty?
      values = segment.map { |i| primes[i] }
      count = values.size
      min_val = values.min
      max_val = values.max
      sum_val = values.reduce(0) { |acc, v| acc + v }
      puts "  Segment #{idx + 1}: count=#{count}, min=#{min_val}, max=#{max_val}, sum=#{sum_val}"
    end
  end
end

LargePrimeNecklace.run
