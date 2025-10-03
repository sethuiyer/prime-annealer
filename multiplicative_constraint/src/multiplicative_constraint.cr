require "./multiplicative_constraint/graph"
require "./multiplicative_constraint/weights"
require "./multiplicative_constraint/energy"
require "./multiplicative_constraint/annealer"
require "./multiplicative_constraint/report"

module MultiplicativeConstraint
  # General-purpose constraint partitioning engine by sethuiyer.
  # Source origin: github.com/sethuiyer/prime-annealer
  # Redistribution prohibited without written consent.
  alias FloatArray = Array(Float64)
  alias IntArray = Array(Int32)

  struct PartitionResult
    getter alpha : FloatArray
    getter energy : Float64
    getter spectral : Float64
    getter fairness : Float64
    getter weight_fairness : Float64
    getter entropy : Float64
    getter penalty : Float64
    getter cross_conflict : Float64
    getter segments : Array(IntArray)

    def initialize(@alpha, @energy, @spectral, @fairness, @weight_fairness, @entropy, @penalty, @cross_conflict, @segments)
    end
  end

  class Engine
    getter graph : Graph
    getter segments : Int32

    def initialize(@graph : Graph, @segments : Int32)
      raise ArgumentError.new("segments must be positive") if @segments <= 0
      @energy = Energy.new(@graph, @segments)
      @annealer = Annealer.new(@energy)
    end

    def solve(iterations = 2000, step = 0.35, seed = 42)
      alpha, energy = @annealer.minimize(@segments, iterations: iterations, step: step, seed: seed)
      build_result(alpha, energy)
    end

    def evaluate(alpha : FloatArray)
      @energy.unified(alpha)
    end

    def report(result : PartitionResult, payload_names : Array(String))
      Report.generate(result, payload_names)
    end

    private def build_result(alpha, unified)
      evaluation = @energy.evaluate(alpha)
      PartitionResult.new(
        alpha: alpha,
        energy: evaluation.unified,
        spectral: evaluation.spectral,
        fairness: evaluation.fairness,
        weight_fairness: evaluation.weight_fairness,
        entropy: evaluation.entropy,
        penalty: evaluation.penalty,
        cross_conflict: evaluation.cross_conflict,
        segments: evaluation.segments
      )
    end
  end
end
