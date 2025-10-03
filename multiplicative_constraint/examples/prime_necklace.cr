require "../src/multiplicative_constraint"

module PrimeNecklaceDemo
  include MultiplicativeConstraint

  PRIME_COUNT = 24
  SEGMENTS = 5
  SEEDS = (3001..3011)

  def self.primes(count : Int32) : Array(Int32)
    limit = (count * 20).to_i
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
    result = Array(Int32).new
    sieve.each_with_index do |flag, idx|
      result << idx if flag
      break if result.size == count
    end
    result
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
      weight = 1.0 + gaps[i]
      adjacency[i][j] = adjacency[j][i] = weight
    end

    n.times do |i|
      j = (i + 2) % n
      longer = 0.5 * (gaps[i] + gaps[(i + 1) % n])
      adjacency[i][j] = adjacency[j][i] = 0.5 * (adjacency[i][j] + longer)
    end

    adjacency
  end

  def self.segment_labels(primes : Array(Int32)) : Array(String)
    primes.map_with_index { |p, idx| "p#{idx + 1}=#{p}" }
  end

  def self.run
    primes = primes(PRIME_COUNT)
    weights = build_weights(primes)
    adjacency = build_adjacency(primes)
    labels = segment_labels(primes)

    runs = Array(NamedTuple(seed: Int32, energy: Float64, result: PartitionResult)).new
    SEEDS.each do |seed|
      graph = Graph.new(weights, adjacency)
      engine = Engine.new(graph, SEGMENTS)
      result = engine.solve(iterations: 2600, step: 0.31, seed: seed)
      runs << {seed: seed, energy: result.energy, result: result}
    end

    best = runs.min_by { |run| run[:energy] }
    unless best
      puts "No partitions generated"
      return
    end

    puts "Prime necklace partition (#{PRIME_COUNT} primes, #{SEGMENTS} segments)"
    runs.sort_by { |run| run[:energy] }.each do |run|
      puts "  seed=#{run[:seed]} -> energy=#{run[:energy]}"
    end
    puts
    puts "Selected best configuration (seed=#{best[:seed]}):"
    puts Report.generate(best[:result], labels)
  end
end

PrimeNecklaceDemo.run
