
using Printf

function putline(args::Vararg{Any})
    s = string(rpad(args[1], 8, ' '))
    s *= join([lpad(arg, 12, ' ') for arg in args[2:end]])
    return s * "\n"
end

function write_cube(data, meta, fname)
    # written on the basis of:
    # https://gist.github.com/aditya95sriram/8d1fccbb91dae93c4edf31cd6a22510f (see function "write_cube")
    #
    # data: volumetric data consisting real values
    # meta: dict containing metadata with following keys
    #     atoms: list of atoms in the form (mass, charge, [position])
    #     org: origin
    #     xvec,yvec,zvec: lattice vector basis
    # fname: filename of cubefile

    open(fname, "w") do cube
        write(cube, "Cubefile created by MagFieldLFT.jl\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        natm = length(meta["atoms"])
        nx, ny, nz = size(data)

        write(cube, putline(natm, meta["org"]...))
        write(cube, putline(nx, meta["xvec"]...))
        write(cube, putline(nx, meta["yvec"]...))
        write(cube, putline(nx, meta["zvec"]...))

        for (atom_mass, atom_charge, atom_pos) in meta["atoms"]
            write(cube, putline(atom_mass, atom_charge, atom_pos...))
        end
        for i in 1:nx
            for j in 1:ny
                for k in 1:nz
                    if (i ≠ 1 || j ≠ 1 || k ≠ 1) && k % 6 == 1
                        write(cube, "\n")
                    end
                    write(cube, " $(@sprintf(" %.5E", data[i, j, k]))")
                end
            end
        end
    end
end


function getline(cube)
    l = split(strip(readline(cube)))
    return parse(Int, l[1]), parse.(Float64, l[2:end])
end

function read_cube(fname)
    # written on the basis of:
    # https://gist.github.com/aditya95sriram/8d1fccbb91dae93c4edf31cd6a22510f (see function "write_cube")
    # see also: write_cube()

    meta = Dict()
    nx, ny, nz = 0, 0, 0
    open(fname, "r") do cube
        readline(cube); readline(cube)  # ignore comments
        natm, meta["org"] = getline(cube)
        nx, meta["xvec"] = getline(cube)
        ny, meta["yvec"] = getline(cube)
        nz, meta["zvec"] = getline(cube)
        meta["atoms"] = [getline(cube) for i in 1:natm]
        data = zeros(Float64, nx*ny*nz)
        idx = 1
        for line in eachline(cube)
            for val in split(line)
                data[idx] = parse(Float64, val)
                idx += 1
            end
        end
    end
    data = reshape(data, (nx * ny * nz))
    return data, meta
end