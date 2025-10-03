module MultiplicativeConstraint
  module Report
    extend self

    def generate(result : PartitionResult, labels : Array(String))
      output = String.build do |io|
        io << "Unified energy: #{result.energy}\n"
        io << "Spectral action: #{result.spectral}\n"
        io << "Fairness energy: #{result.fairness}\n"
        io << "Weight fairness: #{result.weight_fairness}\n"
        io << "Entropy: #{result.entropy}\n"
        io << "Multiplicative penalty: #{result.penalty}\n"
        io << "Cross-conflict weight: #{result.cross_conflict}\n"
        io << "\nSegments:\n"
        result.segments.each_with_index do |segment, idx|
          names = segment.map { |i| labels[i]? || "item_#{i}" }
          io << "  Segment #{idx + 1}: #{names.join(", ")}\n"
        end
      end
      output
    end
  end
end
