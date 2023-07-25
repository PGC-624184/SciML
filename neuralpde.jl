using NeuralPDE, Flux, OptimizationOptimisers

linear(u, p, t) = cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)

chain = Flux.Chain( Dense(1, 16, relu),
                    Dense(16,8,relu),
                    Dense(8, 1))


opt = OptimizationOptimisers.Adam(0.01)
alg = NeuralPDE.NNODE(chain, opt)


sol = solve(prob, NeuralPDE.NNODE(chain, opt), verbose=true, abstol=1f-6, maxiters=5000)
plot(sol.t,sol.u,xlabel="Time",ylabel="u(t)",label="NN Solution")
plot!(sol.t,sin.(2*π*sol.t)./(2π),label="Analytic Solution")