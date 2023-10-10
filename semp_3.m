close all;
% Carregando os dados
load robot_arm.dat
entrada = robot_arm(:,1);
saida = robot_arm(:,2);

% Normalizando os dados de entrada e saída
entrada = (entrada - min(entrada)) / (max(entrada) - min(entrada));
saida = (saida - min(saida)) / (max(saida) - min(saida));

% Dividindo os dados em treinamento e teste
n = length(entrada);
n_train = round(0.2 * n); % 20% dos dados para treinamento

entrada_train = entrada(1:n_train);
saida_train = saida(1:n_train);

entrada_test = entrada(n_train+1:end);
saida_test = saida(n_train+1:end);

erro_depois = [];
saida_prevista = [];


% ordem do modelo SEMP
na = 20; 
nb = 20;

% Definindo um limite para o SRR
limite_SRR = 0.03;
SRR_selecionado = [];
SRR_historico = [];
MSSE_historico = [];

% Preparando a matriz de regressores candidatos
X = [];
for i = max(na,nb)+1:length(entrada_train)
    regressores = [saida_train(i-na:i-1).^2; entrada_train(i-nb:i-1).^2];
    X = [X; regressores'];
end

% Inicializando o modelo vazio (Psi_in) e o modelo candidato cheio (Psi_out)
Psi_in = [];
Psi_out = 1:size(X, 2);
disp(size(Psi_out));

% Inicializando o vetor de erro
erro_vec = zeros(length(entrada_train),1);

% Inicializando o vetor para armazenar o erro de cada iteração
%erro_iter = [];

% Variância da saída
var_saida = var(saida_train);

% Loop para atualizar os modelos
while ~isempty(Psi_out)
    % Inicializando o vetor SRR
    SRR_vec = zeros(length(Psi_out),1);
    i = 1;
    while i <= length(Psi_out)
        % Selecionando um subconjunto de regressores
        X_sub = X(:, Psi_out(i));
        
        % Estimando os parâmetros do modelo usando mínimos quadrados
        b = regress(saida_train(max(na,nb)+1:end), X_sub);
        
        % Calculando a saída prevista para amostra i
        saida_prevista = X(:, Psi_out(i)) * b;
        
        % Calculando o erro
        erro = saida_train(max(na,nb)+1:end) - saida_prevista;
        erro_vec(max(na,nb)+1:end) = erro;

        % Armazenando o erro para esta amostra
        erro_depois = [erro_depois; erro];
        
        % Calculando o MSSE
        MSSE = mean(erro.^2);
        MSSE_historico = [MSSE_historico, MSSE];
        % Armazenando o MSSE para esta iteração
        %erro_iter = [erro_iter; MSSE];
        
        % Calculando a variância da saída
        var_saida = var(saida_train(max(na,nb)+1:end));
        
        % Calculando o SRR
        SRR = 1 - MSSE / var_saida;
        %disp(SRR);

        % Se o SRR for pequeno, remova o regressor do modelo
        if SRR < limite_SRR
            Psi_out(i) = [];
            continue;
        end
      
        % Armazenando o SRR no vetor SRR_vec
        SRR_vec(i) = SRR;

        i = i + 1;
    end
    
    % Encontrando o regressor com o maior SRR
    [~, idx] = max(SRR_vec(1:length(Psi_out)));
        
    % Adicionando o regressor selecionado ao modelo atual P e removendo-o de Psi_out
    Psi_in = [Psi_in, Psi_out(idx)];
    % Armazenando o SRR do regressor selecionado
    SRR_selecionado = [SRR_selecionado, SRR_vec(idx)];
    SRR_historico = [SRR_historico, SRR_vec(idx)];

    Psi_out(idx) = [];
end

% Imprimindo os regressores selecionados e seus respectivos SRRs
fprintf('Os regressores selecionados com o limite_SRR: %.4f são: \n', limite_SRR);
for i = 1:length(Psi_in)
    fprintf('Regressor: %d, SRR: %.4f\n', Psi_in(i), SRR_selecionado(i));
end

% Preparando a matriz de regressores candidatos para os dados de teste
X_test = [];
for i = max(na,nb)+1:length(entrada_test)
    regressores_test = [saida_test(i-na:i-1).^2; entrada_test(i-nb:i-1).^2];
    X_test = [X_test; regressores_test'];
end

% Calculando a saída prevista para os dados de teste
saida_prevista_test = X_test(:, Psi_in) * b;

% Calculando o erro para os dados de teste
erro_test = saida_test(max(na,nb)+1:end) - saida_prevista_test;

% Calculando o MSSE para os dados de teste
MSSE_test = mean(erro_test.^2);

fprintf('O MSSE para os dados de teste é: ');
fprintf('%.4f\n', MSSE_test);

% Plotando a saída real e a saída prevista para os dados de treinamento
figure;
plot(saida_train(max(na,nb)+1:end), 'Color', 'b');
hold on;
plot(saida_prevista, 'Color', [1.0 0.5 0.0]);
xlabel('Amostra');
ylabel('Saída');
legend('Saída Real (Treino)', 'Saída Prevista (Treino)');
title('Comparação da Saída Real e Prevista para os Dados de Treinamento');

% Plotando a saída real e a saída prevista para os dados de teste
figure;
plot(saida_test(max(na,nb)+1:end), 'Color', 'b');
hold on;
plot(saida_prevista_test, 'Color', [1.0 0.5 0.0]);
xlabel('Amostra');
ylabel('Saída');
legend('Saída Real (Teste)', 'Saída Prevista (Teste)');
title('Comparação da Saída Real e Prevista para os Dados de Teste');
