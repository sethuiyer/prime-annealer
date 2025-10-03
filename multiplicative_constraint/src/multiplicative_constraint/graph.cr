module MultiplicativeConstraint
  class Graph
    getter adjacency : Array(Array(Float64))
    getter weights : Array(Float64)
    getter edges : Array(Tuple(Int32, Int32, Float64))

    def initialize(@weights : Array(Float64), @adjacency : Array(Array(Float64)))
      raise ArgumentError.new("weights cannot be empty") if @weights.empty?
      raise ArgumentError.new("adjacency must be square") unless square?(@adjacency)
      raise ArgumentError.new("dimension mismatch") unless @adjacency.size == @weights.size
      @edges = build_edges(@adjacency)
    end

    def size
      @weights.size
    end

    private def square?(matrix)
      matrix.all? { |row| row.size == matrix.size }
    end

    private def build_edges(matrix)
      edges = Array(Tuple(Int32, Int32, Float64)).new
      matrix.size.times do |i|
        ((i + 1)...matrix.size).each do |j|
          weight = matrix[i][j]
          next if weight == 0.0
          edges << {i.to_i32, j.to_i32, weight}
        end
      end
      edges
    end
  end
end
