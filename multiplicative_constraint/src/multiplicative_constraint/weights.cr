module MultiplicativeConstraint
  module Weights
    extend self

    def sieve(limit : Int32)
      raise ArgumentError.new("limit must be >= 2") if limit < 2
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
      weights = Array(Float64).new
      sieve.each_with_index do |flag, idx|
        weights << idx.to_f64 if flag
      end
      weights
    end

    def assign(count : Int32, scale : Array(Float64) = Array(Float64).new)
      pool = sieve(20 * count)
      raise "insufficient weights" if pool.size < count
      base = pool.first(count)
      if scale.empty?
        return base
      end
      raise ArgumentError.new("scale length mismatch") unless scale.size == count
      base.each_with_index.map do |core, idx|
        core * (1.0 + scale[idx])
      end.to_a
    end
  end
end
