using LinearAlgebra
using EvalMetrics
using Images
using MLDatasets

function add_noise_unif(A_p, δ)
    T = typeof(A_p)
    T(clamp.(A_p + rand(-δ:δ, size(A_p)), 0, 255))
end

function add_noise_norm(A_p, σ) 
    T = typeof(A_p)
    T(clamp.(A_p + T(round.(σ * randn(size(A_p)))), 0, 255))
end

function predict()

end


train_imgs, train_labels = MNIST(Tx=Integer, split=:train)[:]
test_imgs, test_labels = MNIST(Tx=Integer, split=:test)[:]




δ = σ = 10
display(clamp.(A, 0, 255))

display(add_noise_unif(A, δ))

display(add_noise_norm(A, σ))

