using Revise

using Random

import Flux
import Zygote

using Plots

Random.seed!(123)


ω = π / 2
X = LinRange(0, 2π, 100)
Y = @. sin(ω * X)

lstm = Flux.f64(Flux.Chain(Flux.LayerNorm(1, 1), Flux.LSTM(1, 32), Flux.Dense(32, 1)))

# Check the results before training.
Y0 = [lstm([x])[1] for x in X]

function loss(model, xs, ys)
    loss = 0.0
    for (x, y) in zip(xs, ys)
        loss += abs2(model([x])[1] - y)
    end
    return loss / length(xs)
end

opt_state = Flux.setup(Flux.Adam(), lstm)

# Training loop.
for epoch = 1:1000
    global lstm
    Flux.reset!(lstm)
    loss_val, grads = Zygote.withgradient(m -> loss(m, X, Y), lstm)
    display(loss_val)
    _, lstm = Flux.update!(opt_state, lstm, grads[1])
end

# Check the results after training.
Flux.reset!(lstm)
Y1 = [lstm([x])[1] for x in X]
begin
    plot(X, Y, label = "True data")
    plot!(X, Y0, label = "Before training")
    plot!(X, Y1, label = "After training")
end
