module KelvinHelmholtzInstability

export simulate

using Plots

@fastmath @inbounds function simulate(resolution=128)
    
    # constants
    N       = resolution
    boxsize = 1.
    γ       = 5/3
    courant = 0.4
    tₛ      = 0
    tₑ      = 2

    # initialize intermediate matrices
    ρ       = Array{Float64}(undef, (N, N))
    νₓ      = Array{Float64}(undef, (N, N))
    νᵧ      = Array{Float64}(undef, (N, N))
    P       = Array{Float64}(undef, (N, N))
    M       = Array{Float64}(undef, (N, N))
    Mₓ      = Array{Float64}(undef, (N, N))
    Mᵧ      = Array{Float64}(undef, (N, N))
    E       = Array{Float64}(undef, (N, N))
    δρδx    = Array{Float64}(undef, (N, N))
    δρδy    = Array{Float64}(undef, (N, N))
    δνₓδx   = Array{Float64}(undef, (N, N))
    δνₓδy   = Array{Float64}(undef, (N, N))
    δνᵧδx   = Array{Float64}(undef, (N, N))
    δνᵧδy   = Array{Float64}(undef, (N, N))
    δPδx    = Array{Float64}(undef, (N, N))
    δPδy    = Array{Float64}(undef, (N, N))
    ρ′      = Array{Float64}(undef, (N, N))
    νₓ′     = Array{Float64}(undef, (N, N))
    νᵧ′     = Array{Float64}(undef, (N, N))
    P′      = Array{Float64}(undef, (N, N))
    ρₓₗ     = Array{Float64}(undef, (N, N))
    ρₓᵣ     = Array{Float64}(undef, (N, N))
    ρᵧₗ     = Array{Float64}(undef, (N, N))
    ρᵧᵣ     = Array{Float64}(undef, (N, N))
    νₓₓₗ    = Array{Float64}(undef, (N, N))
    νₓₓᵣ    = Array{Float64}(undef, (N, N))
    νₓᵧₗ    = Array{Float64}(undef, (N, N))
    νₓᵧᵣ    = Array{Float64}(undef, (N, N))
    νᵧₓₗ    = Array{Float64}(undef, (N, N))
    νᵧₓᵣ    = Array{Float64}(undef, (N, N))
    νᵧᵧₗ    = Array{Float64}(undef, (N, N))
    νᵧᵧᵣ    = Array{Float64}(undef, (N, N))
    Pₓₗ     = Array{Float64}(undef, (N, N))
    Pₓᵣ     = Array{Float64}(undef, (N, N))
    Pᵧₗ     = Array{Float64}(undef, (N, N))
    Pᵧᵣ     = Array{Float64}(undef, (N, N))
    Mfluxₓ  = Array{Float64}(undef, (N, N))
    Mₓfluxₓ = Array{Float64}(undef, (N, N))
    Mᵧfluxₓ = Array{Float64}(undef, (N, N))
    Efluxₓ  = Array{Float64}(undef, (N, N))
    Mfluxᵧ  = Array{Float64}(undef, (N, N))
    Mᵧfluxᵧ = Array{Float64}(undef, (N, N))
    Mₓfluxᵧ = Array{Float64}(undef, (N, N))
    Efluxᵧ  = Array{Float64}(undef, (N, N))
    Eₗ      = Array{Float64}(undef, (N, N))
    Eᵣ      = Array{Float64}(undef, (N, N))
    ρ̄       = Array{Float64}(undef, (N, N))
    M̄ₓ      = Array{Float64}(undef, (N, N))
    M̄ᵧ      = Array{Float64}(undef, (N, N))
    Ē       = Array{Float64}(undef, (N, N))
    P̄       = Array{Float64}(undef, (N, N))
    C       = Array{Float64}(undef, (N, N))

    # grid
    δₓ = boxsize / N
    V  = δₓ^2
    Y  = [x for y in 1:N, x in range(0.5 * δₓ, boxsize - 0.5 * δₓ, N)]
    X  = Y'

    # initial conditions
    # opposite moving streams with perturbation
    ω₀   = 0.1
    σ    = 0.05 / √2
    ρ   .= @. 1 + (abs(Y - 0.5) < 0.25)
    νₓ  .= @. -0.5 + (abs(Y - 0.5) < 0.25)
    νᵧ  .= @. ω₀ * sin(4 * π * X) * (exp(-((Y - 0.25)^2 / (2 * σ^2))) + exp(-((Y - 0.75)^2 / (2 * σ^2))))
    P   .= 2.5
    M   .= @. ρ * V
    Mₓ  .= @. ρ * νₓ * V
    Mᵧ  .= @. ρ * νᵧ * V
    E   .= @. (P / (γ - 1) + 0.5 * ρ * (νₓ^2 + νᵧ^2)) * V
    
    function getFlux!(ρₗ, ρᵣ, νₓₗ, νₓᵣ, νᵧₗ, νᵧᵣ, Pₗ, Pᵣ, Mflux, Mₓflux, Mᵧflux, Eflux, i, j)
        # left and right energies
        Eₗ[i, j] = Pₗ[i, j] / (γ - 1) + 0.5 * ρₗ[i, j] * (νₓₗ[i, j]^2 + νᵧₗ[i, j]^2)
        Eᵣ[i, j] = Pᵣ[i, j] / (γ - 1) + 0.5 * ρᵣ[i, j] * (νₓᵣ[i, j]^2 + νᵧᵣ[i, j]^2)
    
        # compute averaged states
        ρ̄[i, j] = 0.5 * (ρₗ[i, j] + ρᵣ[i, j])
        M̄ₓ[i, j] = 0.5 * (ρₗ[i, j] * νₓₗ[i, j] + ρᵣ[i, j] * νₓᵣ[i, j])
        M̄ᵧ[i, j] = 0.5 * (ρₗ[i, j] * νᵧₗ[i, j] + ρᵣ[i, j] * νᵧᵣ[i, j])
        Ē[i, j]  = 0.5 * (Eₗ[i, j] + Eᵣ[i, j])
        P̄[i, j]  = (γ - 1) * (Ē[i, j] - 0.5 * (M̄ₓ[i, j]^2 + M̄ᵧ[i, j]^2) / ρ̄[i, j])
        
        # compute fluxes (local Lax-Friedrichs/Rusanov)
        Mflux[i, j]  = M̄ₓ[i, j]
        Mₓflux[i, j] = M̄ₓ[i, j]^2 / ρ̄[i, j] + P̄[i, j]
        Mᵧflux[i, j] = M̄ₓ[i, j] * M̄ᵧ[i, j] / ρ̄[i, j]
        Eflux[i, j]  = (Ē[i, j] + P̄[i, j]) * M̄ₓ[i, j] / ρ̄[i, j]
        
        # find wavespeeds
        C[i, j] = max(
            sqrt(γ * Pₗ[i, j] / ρₗ[i, j]) + abs(νₓₗ[i, j]),
            sqrt(γ * Pᵣ[i, j] / ρᵣ[i, j]) + abs(νₓᵣ[i, j])
        )
        
        # add stabilizing diffusive term
        Mflux[i, j]  -= C[i, j] * 0.5 * (ρₗ[i, j] - ρᵣ[i, j])
        Mₓflux[i, j] -= C[i, j] * 0.5 * (ρₗ[i, j] * νₓₗ[i, j] - ρᵣ[i, j] * νₓᵣ[i, j])
        Mᵧflux[i, j] -= C[i, j] * 0.5 * (ρₗ[i, j] * νᵧₗ[i, j] - ρᵣ[i, j] * νᵧᵣ[i, j])
        Eflux[i, j]  -= C[i, j] * 0.5 * (Eₗ[i, j] - Eᵣ[i, j])

    end

    t = tₛ
    while t < tₑ

        Threads.@threads for j in 1:N
            for i in 1:N
                # update primitives
                ρ[i, j]  = M[i, j]  / V
                νₓ[i, j] = Mₓ[i, j] / ρ[i, j] / V
                νᵧ[i, j] = Mᵧ[i, j] / ρ[i, j] / V
                P[i, j]  = (E[i, j] / V - 0.5 * ρ[i, j] * (νₓ[i, j]^2 + νᵧ[i, j]^2)) * (γ - 1)
            end
        end

        # calculate maximum timestep
        δₜ = courant * minimum(@. δₓ / (sqrt( γ * P / ρ ) + sqrt(νₓ^2 + νᵧ^2)))

        Threads.@threads for j in 1:N
            for i in 1:N
                i⁺ = i == N ? 1 : (i + 1)
                i⁻ = i == 1 ? N : (i - 1)
                j⁺ = j == N ? 1 : (j + 1)
                j⁻ = j == 1 ? N : (j - 1)

                # calculate the gradients
                δρδx[i, j] = (ρ[i⁺, j] - ρ[i⁻, j]) / 2 / δₓ
                δρδy[i, j] = (ρ[i, j⁺] - ρ[i, j⁻]) / 2 / δₓ
                δνₓδx[i, j] = (νₓ[i⁺, j] - νₓ[i⁻, j]) / 2 / δₓ
                δνₓδy[i, j] = (νₓ[i, j⁺] - νₓ[i, j⁻]) / 2 / δₓ
                δνᵧδx[i, j] = (νᵧ[i⁺, j] - νᵧ[i⁻, j]) / 2 / δₓ
                δνᵧδy[i, j] = (νᵧ[i, j⁺] - νᵧ[i, j⁻]) / 2 / δₓ
                δPδx[i, j] = (P[i⁺, j] - P[i⁻, j]) / 2 / δₓ
                δPδy[i, j] = (P[i, j⁺] - P[i, j⁻]) / 2 / δₓ

                # extrapolate half time step
                ρ′[i, j]  = ρ[i, j]  - 0.5 * δₜ * (νₓ[i, j] * δρδx[i, j]  + ρ[i, j]  * δνₓδx[i, j] + νᵧ[i, j] * δρδy[i, j] + ρ[i, j] * δνᵧδy[i, j])
                νₓ′[i, j] = νₓ[i, j] - 0.5 * δₜ * (νₓ[i, j] * δνₓδx[i, j] + νᵧ[i, j] * δνₓδy[i, j] + (1 / ρ[i, j]) * δPδx[i, j])
                νᵧ′[i, j] = νᵧ[i, j] - 0.5 * δₜ * (νₓ[i, j] * δνᵧδx[i, j] + νᵧ[i, j] * δνᵧδy[i, j] + (1 / ρ[i, j]) * δPδy[i, j])
                P′[i, j]  = P[i, j]  - 0.5 * δₜ * (γ * P[i, j] * (δνₓδx[i, j] + δνᵧδy[i, j]) + νₓ[i, j] * δPδx[i, j] + νᵧ[i, j] * δPδy[i, j])

                # extrapolate in space to face centers
                ρₓₗ[i⁻, j] = ρ′[i, j] - (δρδx[i, j] * δₓ / 2)
                ρₓᵣ[i, j]  = ρ′[i, j] + (δρδx[i, j] * δₓ / 2)
                ρᵧₗ[i, j⁻] = ρ′[i, j] - (δρδy[i, j] * δₓ / 2)
                ρᵧᵣ[i, j]  = ρ′[i, j] + (δρδy[i, j] * δₓ / 2)
                νₓₓₗ[i⁻, j] = νₓ′[i, j] - (δνₓδx[i, j] * δₓ / 2)
                νₓₓᵣ[i, j]  = νₓ′[i, j] + (δνₓδx[i, j] * δₓ / 2)
                νₓᵧₗ[i, j⁻] = νₓ′[i, j] - (δνₓδy[i, j] * δₓ / 2)
                νₓᵧᵣ[i, j]  = νₓ′[i, j] + (δνₓδy[i, j] * δₓ / 2)
                νᵧₓₗ[i⁻, j] = νᵧ′[i, j] - (δνᵧδx[i, j] * δₓ / 2)
                νᵧₓᵣ[i, j]  = νᵧ′[i, j] + (δνᵧδx[i, j] * δₓ / 2)
                νᵧᵧₗ[i, j⁻] = νᵧ′[i, j] - (δνᵧδy[i, j] * δₓ / 2)
                νᵧᵧᵣ[i, j]  = νᵧ′[i, j] + (δνᵧδy[i, j] * δₓ / 2)
                Pₓₗ[i⁻, j] = P′[i, j] - (δPδx[i, j] * δₓ / 2)
                Pₓᵣ[i, j]  = P′[i, j] + (δPδx[i, j] * δₓ / 2)
                Pᵧₗ[i, j⁻] = P′[i, j] - (δPδy[i, j] * δₓ / 2)
                Pᵧᵣ[i, j]  = P′[i, j] + (δPδy[i, j] * δₓ / 2)
            end
        end

        Threads.@threads for j in 1:N
            for i in 1:N
                # compute fluxes (local Lax-Friedrichs/Rusanov)
                getFlux!(ρₓₗ, ρₓᵣ, νₓₓₗ, νₓₓᵣ, νᵧₓₗ, νᵧₓᵣ, Pₓₗ, Pₓᵣ, Mfluxₓ, Mₓfluxₓ, Mᵧfluxₓ, Efluxₓ, i, j)
                getFlux!(ρᵧₗ, ρᵧᵣ, νᵧᵧₗ, νᵧᵧᵣ, νₓᵧₗ, νₓᵧᵣ, Pᵧₗ, Pᵧᵣ, Mfluxᵧ, Mᵧfluxᵧ, Mₓfluxᵧ, Efluxᵧ, i, j)
            end
        end

        Threads.@threads for j in 1:N
            for i in 1:N
                i⁻ = i == 1 ? N : (i - 1)
                j⁻ = j == 1 ? N : (j - 1)
            
                # update solution
                M[i, j]  =  M[i, j] + δₜ * δₓ * ( -Mfluxₓ[i, j] +  Mfluxₓ[i⁻, j] -  Mfluxᵧ[i, j] +  Mfluxᵧ[i, j⁻])
                Mₓ[i, j] = Mₓ[i, j] + δₜ * δₓ * (-Mₓfluxₓ[i, j] + Mₓfluxₓ[i⁻, j] - Mₓfluxᵧ[i, j] + Mₓfluxᵧ[i, j⁻])
                Mᵧ[i, j] = Mᵧ[i, j] + δₜ * δₓ * (-Mᵧfluxₓ[i, j] + Mᵧfluxₓ[i⁻, j] - Mᵧfluxᵧ[i, j] + Mᵧfluxᵧ[i, j⁻])
                E[i, j]  =  E[i, j] + δₜ * δₓ * ( -Efluxₓ[i, j] +  Efluxₓ[i⁻, j] -  Efluxᵧ[i, j] +  Efluxᵧ[i, j⁻])
            end
        end

        t += δₜ
        
    end

    heatmap(ρ)

end

end