function out = normalize(X)

  mean = mean(X,1);

  std = std(X,1);

  out = (X .- mean) ./ std;

  % disp(size(X))
  % disp(size(out))
end
