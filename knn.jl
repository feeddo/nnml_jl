using LinearAlgebra
using Images
using MLDatasets
using TensorCast
using Base.Sort
using StatsBase
using StatisticalMeasures
using GraphSignals, Distances, Graphs

function add_noise_unif(A_p, δ)
    T = typeof(A_p)
    T(clamp.(A_p + rand(-δ:δ, size(A_p)), 0, 255))
end

function add_noise_norm(A_p, σ) 
    T = typeof(A_p)
    T(clamp.(A_p + T(round.(σ * randn(size(A_p)))), 0, 255))
end

function predict(train_img_vecs, train_labels, test_img_vecs, k)
    train_count = size(train_img_vecs, 2)
    test_count = size(test_img_vecs, 2)
    test_labels_pred = Array{Int}(undef, test_count)
    for i = 1:test_count
        distances = tuple.(norm.(eachcol(train_img_vecs .- test_img_vecs[:,i])), train_labels)
        k_smallest_dists = sort(distances; alg=Sort.PartialQuickSort(k))[1:k] 
        test_labels_pred[i] = mode([ x[2] for x in k_smallest_dists ])
    end
    test_labels_pred
end

function detect_outliers_type_2(train_img_vecs)
    train_img_vecs_transp = float(train_img_vecs)
    k = 7
    graph = kneighbors_graph(train_img_vecs_transp, k)
    edge_dests = countmap(collect(edges(graph))[2])
    print("outliers of the 2nd type:\n")
    for i in eachindex(test_labels)
        if !haskey(edge_dests, i)
            print("#", i, " is not a neighbor for any point\n")
        elseif edge_dests[i] < 3
            print("#", i, " is a neighbor for only ", edge_dests[i], " points\n")
        end
    end
    print("\n")
end


train_imgs, train_labels = MNIST(Tx=Int16, split=:train)[1:500]
test_imgs, test_labels = MNIST(Tx=Int16, split=:test)[1:100]

# 28x28xN -> 784xN
@cast train_img_vecs[(i,j), k] := train_imgs[i,j,k];
@cast test_img_vecs[(i,j), k] := test_imgs[i,j,k];

test_img_vecs_nu = add_noise_unif(test_img_vecs, 100)
test_img_vecs_nn = add_noise_norm(test_img_vecs, 100)

detect_outliers_type_2(train_img_vecs)

for k = 1:20
    test_labels_pred = predict(train_img_vecs, train_labels, test_img_vecs, k)
    test_labels_pred_nu = predict(train_img_vecs, train_labels, test_img_vecs_nu, k)
    test_labels_pred_nn = predict(train_img_vecs, train_labels, test_img_vecs_nn, k)
    printstyled("k = ", k, "\n"; color = :yellow)
    print("accuracy: ", accuracy(test_labels_pred, test_labels), ", F1 score: ", multiclass_f1score(test_labels_pred, test_labels), "\n")
    print("(unif noise) accuracy: ", accuracy(test_labels_pred_nu, test_labels), ", F1 score: ", multiclass_f1score(test_labels_pred_nu, test_labels), "\n")
    print("(norm noise) accuracy: ", accuracy(test_labels_pred_nn, test_labels), ", F1 score: ", multiclass_f1score(test_labels_pred_nn, test_labels), "\n")
    
    print("outliers of the 1st type:\n")
    for i in eachindex(test_labels)
        if test_labels[i] != test_labels_pred[i]
            print("[#", i, " ")
            printstyled(test_labels_pred[i], " "; color = :red)
            printstyled(test_labels[i], color = :green)
            print("] ")
        end
    end
    print("\n\n")
end

