function [J grad] = cost_grad_nn(nn_params, ...
                                 input_layer_size, ...
                                 hidden_layer_size, ...
                                 num_labels, ...
                                 X, y, lambda)
% nn_params should be a 3-dim matrix
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

A1 = [ones(m,1) X];
Z2 = A1 * Theta1';
A2 = [ones(m,1) sigmoid(Z2)];
Z3 = A2 * Theta2';
A3 = Z3;

J = 1/(2*m) * sum((A3 - y) .^ 2) + lambda/(2*m);

for i = 1:m
  delta_3 = A3(i,:)' - y(i,:);
  delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(Z2(i,:)')];

  Theta1_grad = Theta1_grad + delta_2(2:end) * A1(i,:);
  Theta2_grad = Theta2_grad + delta_3 * A2(i,:);
end

J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
disp(J)
Theta1_reg_term = [zeros(size(Theta1,1), 1) lambda/m * Theta1(:, 2:end)];
Theta2_reg_term = [zeros(size(Theta2,1), 1) lambda/m * Theta2(:, 2:end)];
Theta1_grad = 1/m * Theta1_grad + Theta1_reg_term;
Theta2_grad = 1/m * Theta2_grad + Theta2_reg_term;


% % J = (1/m) * sum(-y .* log(A3') - (1-y) .* log(1 - A3'));
% J = 1/(2*m) * sum((A3 - y) .^ 2) + lambda/(2*m);
%
% delta_3 = A3 - y;
% delta_2 = (Theta2' .* delta_3') .* [1; sigmoidGradient(Z2')];
%
% Theta1_grad = Theta1_grad + delta_2(2:end) * A1;
% Theta2_grad = Theta2_grad + delta_3 * A2;
%
%
% J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
%
% Theta1_reg_term = [zeros(size(Theta1,1), 1) lambda/m * Theta1(:, 2:end)];
% Theta2_reg_term = [zeros(size(Theta2,1), 1) lambda/m * Theta2(:, 2:end)];
% Theta1_grad = 1/m * Theta1_grad + Theta1_reg_term;
% Theta2_grad = 1/m * Theta2_grad + Theta2_reg_term;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
