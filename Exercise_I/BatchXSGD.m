clear all
clc
X = [0 1 1 1 1 1 1 1;
   0 0 0 1 1 0 0 1;
   1 0 1 1 0 1 1 1;
   1 0 1 1 1 1 0 1;
   1 1 0 1 1 0 0 1;
   1 1 1 0 1 1 1 1;
   1 1 1 0 1 1 1 1;
   0 0 1 1 1 0 0 1;
   1 1 1 1 1 1 1 1;
   1 1 1 1 1 0 0 1];

D = [0 0 0 0 0 0 0 0 0;
   1 0 0 0 0 0 0 0 0 ;
   0 1 0 0 0 0 0 0 0;
   0 0 1 0 0 0 0 0 0;
   0 0 0 1 0 0 0 0 0;
   0 0 0 0 1 0 0 0 0;
   0 0 0 0 0 1 0 0 0 ;
   0 0 0 0 0 0 1 0 0;
   0 0 0 0 0 0 0 1 0;
   0 0 0 0 0 0 0 0 1];


E1 = zeros(1000, 1);
E2 = zeros(1000, 1);
W1 = 2*rand(9, 8) - 1;
W2 = W1;
B1 = 2*rand(9,1) - 1;
B2 = B1;
for epoch = 1:1000 % train
   W1 = DeltaSGD(W1, B1, X, D);
   W2 = DeltaBatch(W2, B2, X, D);
   es1 = 0;
   es2 = 0;
   N = size(X,1);
   for k = 1:N
       x = X(k, :)';
       d = D(k);
      
       v1 = W1*x;
       y1 = Sigmoid(v1);
       es1 = es1 + (d(1) - y1(1))^2;
      
       v2 = W2*x;
       y2 = Sigmoid(v2);
       es2 = es2 + (d(1) - y2(1))^2;
   end
   E1(epoch) = es1/N;
   E2(epoch) = es2/N;
end
% Média dos erros finais
meanErrorSGD = mean(E1);
meanErrorBatch = mean(E2);
% Exibir médias dos erros
fprintf('Média do erro de treinamento para SGD: %.4f\n', meanErrorSGD);
fprintf('Média do erro de treinamento para Batch Gradient Descent: %.4f\n', meanErrorBatch);
% Comparar valores desejados e previstos
disp('Comparação dos valores desejados e previstos:');
disp('Índice | Saída Desejada | Previsão SGD | Previsão Batch');
disp('--------------------------------------------------------');
for k = 1:size(X, 1)
   x = X(k, :)';
   d = D(k, :)'; % Saída desejada
  
   % Previsões com pesos treinados
   v1 = W1 * x;
   y1 = Sigmoid(v1);
  
   v2 = W2 * x;
   y2 = Sigmoid(v2);
  
   % Exibir resultados
   fprintf('%d      | %s      | %s      | %s\n', ...
       k, ...
       mat2str(d', 4), ... % Saída desejada
       mat2str(y1, 4), ... % Previsão SGD
       mat2str(y2, 4));   % Previsão Batch
end
% Exibir pesos finais
disp('Pesos finais da rede treinada:');
disp('Método   | Pesos');
disp('-------------------');
disp(['SGD      | ' mat2str(W1, 4)]);
disp(['Batch    | ' mat2str(W2, 4)]);
% Exibir tabela de entrada de dados
disp('Tabela de Entrada de Dados (0 = Apagado, 1 = Aceso):');
disp(array2table(X, 'VariableNames', arrayfun(@(i) sprintf('Segmento%d', i), 1:size(X, 2), 'UniformOutput', false)));
plot(E1, 'r', 'LineWidth', 1.5)
hold on
plot(E2, 'b:', 'LineWidth', 1.5)
xlabel('Epoch')
ylabel('Average of Training error')
legend('SGD', 'Batch')


function W = DeltaSGD(W, B, X, D)
   alpha = 0.9;
  
   N = size(X,1);
   for k = 1:N
       x = X(k, :)';   %inputs
       d = D(k);       %desired outputs
      
       v = W*x;
       y = Sigmoid(v); %obtained outputs
      
       e = d - y;
       delta = y.*(ones(9,1)-y).*e;
      
       dW = alpha.*(delta*x'); %delta rule
       dB= alpha*e;
       W= W+ dW;
       B=B+dB;
   end
end

function W = DeltaBatch(W, B, X, D)
   alpha = 0.9;
  
   dWsum = zeros(9, 8);
   dBsum =zeros(9, 1);
  
   N = size(X,1);
   for k = 1:N
       x = X(k, :)';
       d = D(k);
       v = W*x;
       y = Sigmoid(v);
       e = d - y;
       delta = y.*(ones(9,1)-y).*e;
       dW = alpha*delta*x';
       dWsum = dWsum + dW;
       dB = alpha*e;
       dBsum = dBsum + dB;
   end
   dWavg = dWsum/N;
   dBavg = dBsum/N;
   W= W+ dWavg;
   B= B+ dBavg;
end

function y = Sigmoid(x)
   y = 1 ./ (1 + exp(-x));
  
end