module MultiplicativeConstraint
  class Annealer
    def initialize(@energy : Energy)
    end

    def minimize(blocks : Int32, iterations = 1500, step = 0.35, seed = 42)
      rng = Random.new(seed)
      alpha = Array.new(blocks) { rng.rand * 2 * Math::PI }
      energy_val = @energy.unified(alpha)
      best_alpha = alpha.dup
      best_energy = energy_val
      step_scale = step
      iterations.times do |iter|
        temperature = Math.max(0.02, 1.0 - iter / iterations.to_f64)
        candidate = alpha.map do |a|
          delta = gaussian(rng, step_scale * temperature)
          (a + delta) % (2 * Math::PI)
        end
        candidate_energy = @energy.unified(candidate)
        if candidate_energy < energy_val || rng.rand < Math.exp(-(candidate_energy - energy_val) / temperature)
          alpha = candidate
          energy_val = candidate_energy
          if candidate_energy < best_energy
            best_energy = candidate_energy
            best_alpha = candidate.dup
          end
        end
        step_scale = Math.max(0.05, step_scale * 0.999)
      end
      {best_alpha, best_energy}
    end

    private def gaussian(rng : Random, scale : Float64)
      u1 = rng.rand.clamp(1e-9, 1.0)
      u2 = rng.rand
      magnitude = Math.sqrt(-2.0 * Math.log(u1))
      angle = 2.0 * Math::PI * u2
      magnitude * Math.cos(angle) * scale
    end
  end
end
