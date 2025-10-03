require "../src/multiplicative_constraint"

module SocialNetworkDemo
  include MultiplicativeConstraint

  struct Profile
    getter name : String
    getter daily_posts : Float64
    getter interactions : Float64
    getter communities : Array(String)

    def initialize(@name, @daily_posts, @interactions, communities : Array(String))
      @communities = communities
    end
  end

  PROFILES = [
    Profile.new("CreatorA", 8.5, 420.0, ["gaming", "streaming", "esports"]),
    Profile.new("CreatorB", 6.2, 310.0, ["streaming", "music"]),
    Profile.new("AnalystC", 4.8, 260.0, ["finance", "tech", "startups"]),
    Profile.new("EngineerD", 3.5, 180.0, ["tech", "opensource"]),
    Profile.new("DesignerE", 5.1, 205.0, ["design", "art", "tech"]),
    Profile.new("PhotographerF", 7.0, 275.0, ["art", "travel"]),
    Profile.new("CommunityG", 2.4, 520.0, ["gaming", "tech", "opensource"]),
    Profile.new("FounderH", 3.1, 330.0, ["startups", "finance", "community"]),
    Profile.new("EducatorI", 4.0, 295.0, ["education", "tech", "community"]),
    Profile.new("ResearcherJ", 2.8, 150.0, ["science", "tech", "opensource"]),
  ]

  def self.normalised(values : Array(Float64))
    max = values.max
    return Array.new(values.size, 0.0) if max <= 0.0
    values.map { |v| v / max }
  end

  def self.build_weights
    posts = normalised(PROFILES.map(&.daily_posts))
    engagements = normalised(PROFILES.map(&.interactions))
    scale = posts.zip(engagements).map do |post_scale, engage_scale|
      0.6 * post_scale + 0.4 * engage_scale
    end
    Weights.assign(PROFILES.size, scale)
  end

  def self.build_adjacency
    n = PROFILES.size
    adjacency = Array.new(n) { Array.new(n, 0.0) }
    n.times do |i|
      ((i + 1)...n).each do |j|
        pi = PROFILES[i]
        pj = PROFILES[j]
        shared = (pi.communities & pj.communities).size
        next if shared.zero?

        post_gap = (pi.daily_posts - pj.daily_posts).abs / (pi.daily_posts + pj.daily_posts)
        interact_gap = (pi.interactions - pj.interactions).abs / (pi.interactions + pj.interactions)
        affinity = shared.to_f + 0.5 * (2.0 - post_gap - interact_gap)
        adjacency[i][j] = adjacency[j][i] = affinity
      end
    end
    adjacency
  end

  def self.run
    weights = build_weights
    adjacency = build_adjacency
    seeds = (2022..2032).to_a

    runs = Array(NamedTuple(seed: Int32, energy: Float64, result: PartitionResult)).new

    seeds.each do |seed|
      graph = Graph.new(weights, adjacency)
      engine = Engine.new(graph, 3)
      result = engine.solve(iterations: 2200, step: 0.32, seed: seed)
      runs << {seed: seed, energy: result.energy, result: result}
    end

    best = runs.min_by { |run| run[:energy] }
    unless best
      puts "No runs executed"
      return
    end

    puts "Seed sweep energy summary:"
    runs.sort_by { |run| run[:energy] }.each do |run|
      puts "  seed=#{run[:seed]} -> energy=#{run[:energy]}"
    end

    puts "\nSelected best configuration (seed=#{best[:seed]}):"
    puts Report.generate(best[:result], PROFILES.map(&.name))
  end
end

SocialNetworkDemo.run
