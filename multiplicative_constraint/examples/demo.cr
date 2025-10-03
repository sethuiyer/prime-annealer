require "../src/multiplicative_constraint"

module Demo
  include MultiplicativeConstraint

  GRAPH_WEIGHTS = [12.0, 15.0, 17.0, 10.0, 8.0, 22.0]
  GRAPH_ADJ = [
    [0.0, 2.0, 1.0, 0.0, 0.0, 4.0],
    [2.0, 0.0, 3.5, 1.0, 0.0, 2.0],
    [1.0, 3.5, 0.0, 0.0, 1.0, 0.5],
    [0.0, 1.0, 0.0, 0.0, 2.3, 0.0],
    [0.0, 0.0, 1.0, 2.3, 0.0, 1.2],
    [4.0, 2.0, 0.5, 0.0, 1.2, 0.0],
  ]
  LABELS = ["TaskA", "TaskB", "TaskC", "TaskD", "TaskE", "TaskF"]

  def self.run
    graph = MultiplicativeConstraint::Graph.new(GRAPH_WEIGHTS, GRAPH_ADJ)
    engine = MultiplicativeConstraint::Engine.new(graph, 3)
    result = engine.solve(iterations: 1500, step: 0.35, seed: 2025)
    puts MultiplicativeConstraint::Report.generate(result, LABELS)
  end
end

Demo.run
