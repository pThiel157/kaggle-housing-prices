function [J, grad] = cost_grad(theta, X, y, lambda)

  m = length(y);
  grad = zeros(size(theta, 1), 1);

  J = 1/(2*m) * sum((X * theta - y) .^ 2) + lambda/(2*m) * sum(theta(2:end).^2);

  grad(1)     = 1/m * sum(X * theta - y)';
  grad(2:end) = 1/m * sum((X * theta - y) .* X(:,2:end))' + lambda/m * theta(2:end);

end
