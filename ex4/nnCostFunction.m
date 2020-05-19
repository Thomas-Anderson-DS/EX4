function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
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
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% y = [y zeros(m,num_labels-1)];
% for i=1:num_labels
%     Tem3 = y((i-1)*m/num_labels+1,1);
%     for j=1:m/num_labels
%         y((i-1)*m/num_labels+j,1) = 0;
%         y((i-1)*m/num_labels+j,i) = Tem3;
%     end
% end
% for i=1:num_labels
%     for j=1:m
%        if(y(j,i) ~= 0)
%             y(j,i) = 1;
%        end
%     end
% end

% fprintf(' %f ',size(X));
% fprintf(' %f ',size(Theta1));
% fprintf(' %f ',size(Theta2));
% fprintf(size(Theta2' * sigmoid(Theta1' * X(1,:))));
X = [zeros(m,1)+1 X];%X means a1_lack
z2 = Theta1 * X';
a2_lack = sigmoid(z2);
a2 = [zeros(1,m)+1;a2_lack];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
%kkk=zeros(10,m);
for i=1:m
    Tem_y = [zeros(1,y(i)-1) 1 zeros(1,num_labels-y(i))];
    J = J - Tem_y * log(a3(:,i)) - (1-Tem_y) * log(1-a3(:,i));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Delta3(:,i) = a3(:,i) - Tem_y';
    
end

Tem_Theta2 = Theta2;
Tem_Theta2(:,1) = [];

Delta2 = Tem_Theta2' * Delta3 .* sigmoidGradient(z2);
% fprintf(' %f ',size(Delta2));
% fprintf(' %f ',size(X));
% fprintf(' %f ',size(Delta3));
% fprintf(' %f ',size(a2));
Theta1_grad = Delta2 * X / m;
Theta2_grad = Delta3 * a2' / m;
% fprintf(' %f ',size(Delta2));
% fprintf(' %f ',size(Delta3));
% fprintf(' %f ',size(X));
% fprintf(' %f ',size(a2));
J = J / m;
%fprintf(' %f ',size(Theta1));
%fprintf(' %f ',size(Theta2));
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;
Theta1(:,1) = [];
Theta2(:,1) = [];
J = J + lambda / 2 / m * ( sum(sum(Theta1.*Theta1,1),2) + sum(sum(Theta2.*Theta2,1),2) );












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
