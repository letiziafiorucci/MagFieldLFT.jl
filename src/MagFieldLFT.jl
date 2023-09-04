module MagFieldLFT

export iscanonical, create_SDs

function iscanonical(orblist::Vector{T}) where T <: Int
    for i in 2:length(orblist)
        if orblist[i]<=orblist[i-1]
            return false
        end
    end
    return true
end

function create_SDs(nel::Int, norb::Int)
    SDs = Vector{Vector{Int64}}()
    nspinorb = 2*norb
    spinorbrange = 1:nspinorb
    allranges = Tuple(repeat([spinorbrange], nel))
    for indices in CartesianIndices(allranges)
        orblist = reverse(collect(Tuple(indices)))
        if iscanonical(orblist)
            push!(SDs, orblist)
        end
    end
    return SDs
end

end