d = csvread("data/train.csv");
d = d(2:end,:);

y = d(:,81);

% Get the data we want:
lotArea = d(:,5);
overallQual = d(:,18);
overallCond = d(:,19);
livArea = d(:,47);
monthSold = d(:,77);
yearSold = d(:,78);

% Set up X and initial theta:
X = [lotArea overallQual overallCond livArea monthSold yearSold];
X = [X (X.^2)];
X = [X lotArea.*overallQual lotArea.*overallCond lotArea.*livArea lotArea.*monthSold lotArea.*yearSold];
X = [X overallQual.*overallCond overallQual.*livArea overallQual.*monthSold overallQual.*yearSold];
X = [X overallCond.*livArea overallCond.*monthSold overallCond.*yearSold];
X = [X livArea.*monthSold livArea.*yearSold];
X = [X monthSold.*yearSold];
X = normalize(X);
X = [ones(size(X,1), 1) X];
initial_theta = zeros(size(X,2), 1);


% Set up neural net stuff if we want to:
% X = X(:, 2:end); % Get rid of the bias column since we'll add that in later
% input_layer_size  = size(X, 2);
% hidden_layer_size = 25;          % 25 hidden units
% cost_grad = @(nn_params, X, y, lambda) cost_grad_nn(nn_params, input_layer_size, hidden_layer_size, 1, X, y, lambda);
% initial_theta = [rand((1 + input_layer_size) * hidden_layer_size, 1) ; rand(hidden_layer_size + 1, 1)]; % Theta1 and Theta2


% Split X and y into train and cross-verification sets:
delim = round((0.8 * size(X,1)));

X_train = X(1:delim, :);
X_cv    = X(delim:end, :);

y_train = y(1:delim);
y_cv    = y(delim:end);

m = size(X, 1);
m_train = size(X_train, 1);
m_cv = size(X_cv, 1);


% Set lambda:
lambda = 500;


% A quick gradient check for sanity:
[_, grad] = cost_grad(initial_theta, X_train, y_train, lambda);
costFunc = @(p) cost_grad(p, X_train, y_train, lambda);
numGrad = computeNumericalGradient(costFunc, initial_theta);
disp([grad numGrad])
disp('The above should be similar')

% Run fminunc (optimize the cost func):
theta = trainLinearReg(X_train, y_train, lambda);

% Another quick gradient check for sanity:
[_, grad] = cost_grad(theta, X_train, y_train, lambda);
costFunc = @(p) cost_grad(p, X_train, y_train, lambda);
numGrad = computeNumericalGradient(costFunc, theta);
disp([grad numGrad])
disp('The above should also be similar')

disp('Theta after optimization:')
disp(theta)

cost_train = cost_grad(theta, X_train, y_train, 0);
disp('Cost after optimization:')
disp(cost_train)

% Run on our cross-verification set and get the cost:
cost_cv = cost_grad(theta, X_cv, y_cv, 0);

disp('CV cost:')
disp(cost_cv)


% Plot learning curve stuff:
max_m = 50;
[error_train, error_val] = learningCurve(X_train, y_train, X_cv, y_cv, lambda, max_m);
plot(1:max_m, error_train, 1:max_m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 max_m 0 4*max(cost_train, cost_cv)])


% Make our predictions for the test set and output a csv file:
test_data = csvread("data/test.csv");
ids = test_data(:,1);

lotArea = test_data(:,5);
overallQual = test_data(:,18);
overallCond = test_data(:,19);
livArea = test_data(:,47);
monthSold = test_data(:,77);
yearSold = test_data(:,78);

test = [lotArea overallQual overallCond livArea monthSold yearSold];
test = [test (test.^2)];
test = [test lotArea.*overallQual lotArea.*overallCond lotArea.*livArea lotArea.*monthSold lotArea.*yearSold];
test = [test overallQual.*overallCond overallQual.*livArea overallQual.*monthSold overallQual.*yearSold];
test = [test overallCond.*livArea overallCond.*monthSold overallCond.*yearSold];
test = [test livArea.*monthSold livArea.*yearSold];
test = [test monthSold.*yearSold];
test = normalize(test);
test = [ones(size(test,1), 1) test];

p = predict(test, theta);

out = [ids p];

csvwrite("out.csv", out);
