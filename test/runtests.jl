using SubsetSelection
using StatsBase
using Base.Test

n = 500; p = 1000; k = 10;

indices = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
w = sample(-1:2:1, k);
X = randn(n,p); Y = X[:,indices]*w;
Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X, averaging=false, gamma=0.1)
for i in 1:k
  @test Sparse_Regressor.indices[i]==indices[i]
end
 @test isapprox(Sparse_Regressor.w, w, atol=0.1)
