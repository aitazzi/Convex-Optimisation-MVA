%% DM Optimisation Abderrahim AIT AZZI
%% Exercice 1
 % Matlab Implementation
Q=10; p=1; A=2; b=-1; % Declare some parameters
t =0.001; % Set the barrier parameter
f = @( x,t ) phi (x , t ,Q, p ,A, b) ;
g = @( x,t) grad (x , t ,Q, p ,A, b) ; 
h = @( x,t ) hessian (x , t ,Q, p ,A, b) ; 
%% Exercice 2
%% Damped Newton method
x0=-1;
tol=0.000001;
[xstar,xhist,Gap] = dampedNewton(x0,t,f,g,h, tol);
%% Back-Linesearch Newton method
x0=-1;
tol=0.000001;
[xstar2,xhist2,Gap2] = newtonLS(x0,t,f,g,h,A,b,tol);
%% Plot \phi_t(x_k)
F2=zeros(length(xhist2),1);
dif2=zeros(length(xhist2),1);
for i=1:length(xhist2)
    F2(i)=f(xhist2(i),t);
end
F=zeros(length(xhist),1);
dif=zeros(length(xhist),1);
for i=1:length(xhist)
    F(i)=f(xhist(i),t);
    dif(i)=f(xhist(i),t)-f(xstar,t);
end
figure()
set(gcf,'color','w')
hold on
plot(F,'b','linewidth',2)
hold on
plot(F2,'r','linewidth',2)
grid on
legend('Damped Newton','Newton LS')
title('Objective function \phi_t (x_k) w.r.t k')
ylabel('\phi_t(x_k)')
xlabel('iterrations')
%% The Estimated Gap
figure()
set(gcf,'color','w')
plot(Gap(1:max(length(Gap),length(Gap2))),'b','linewidth',2)
hold on
plot(Gap2(1:min(length(Gap),length(Gap2))),'r','linewidth',2)
legend('damped Newton', 'LS Newton')
ylabel('\phi_t(x_k)-\phi_t^*')
xlabel('iterrations k')
title('Estimated gap \phi_t(x_k)-\phi_t^*')
%% Exercice 3
%% Support Vector Machine Problem
%% Constructing the data (Iris-versicolor versus Iris-virginica)
load fisheriris
id1=find(strcmp(species, 'versicolor')==1);
id2=find(strcmp(species, 'virginica')==1);
data1=meas(id1,:);
data2=meas(id2,:);
% The training data
X_train=[data1(1:0.8*length(data1),:);data2(1:0.8*length(data2),:)];
X_train=X_train-mean(X_train);
y_train=[ones(0.8*length(data1),1);2*ones(0.8*length(data2),1)];
% The testing data
X_test=[data1(1+0.8*length(data1):end,:);data2(0.8*length(data2)+1:end,:)];
X_test=X_test-mean(X_test);
y_test=[ones(0.2*length(data1),1);2*ones(0.2*length(data2),1)];
%% Plot the training data
id1_train=find(y_train==1); id2_train=find(y_train==2); % The index of the true training labels
figure()
set(gcf,'color','w')
plot(X_train(id1_train,3),X_train(id1_train,4),'ro')
hold on
plot(X_train(id2_train,3),X_train(id2_train,4),'b*')
legend('versicolor','virginica')
title('Data from iris dataset')
grid on
xlabel('petal length')
ylabel('petal width')
%% Barrier Mathod: With the primal problem
tau=0.001;  % The regulazation parameter
[Q_primal,p_primal,A_primal,b_primal] = transform_svm_primal(tau,X_train,y_train);
[n,d]=size(X_train);
x0_primal=[zeros(d,1);2.*ones(n,1)];
mu=100;
tol=0.0000001;
[X_primal,~,~] = barr_method(Q_primal,p_primal,A_primal,b_primal,x0_primal,mu,tol);
%% Barrier Mathod: With the dual problem
tau=0.001; % The regulazation parameter
[Q_dual,p_dual,A_dual,b_dual] = transform_svm_dual(tau,X_train,y_train);
[n,d]=size(X_train);
x0_dual=(1/(2*tau*n)).*ones(n,1);
mu=20;
tol=0.000001;
[X_dual,~] = barr_method(Q_dual,p_dual,A_dual,b_dual,x0_dual,mu,tol);
%% Plot the separator
W=X_primal(1:4);
W_dual=X_train'*diag(y_train)*X_dual; % You can check if w=w_dual
idx1=3;idx2=4; % Choose the 2 attributes that we want to display
id_pred1 = find(X_train(:,[idx1,idx2])*W([idx1,idx2])< 0);
id_pred2 = find(X_train(:,[idx1,idx2])*W([idx1,idx2])>= 0);
figure()
set(gcf,'color','w')
plot(X_train(id1_train,idx1),X_train(id1_train,idx2),'ro')
hold on
plot(X_train(id_pred1,idx1),X_train(id_pred1,idx2),'rx')
hold on
plot(X_train(id2_train,idx1),X_train(id2_train,idx2),'bo')
hold on
plot(X_train(id_pred2,idx1),X_train(id_pred2,idx2),'bx')
hold on
grid on
x=min(X_train(:,idx1))-0.5:0.1:max(X_train(:,idx1))+0.5;
plot(x,(-W(idx1)/W(idx2))*x,'g','linewidth',2)
legend('true versicolor', 'pred versicolor ','true virginica', 'pred virginica','SVM Boundary')
axis([min(X_train(:,idx1))-0.5 max(X_train(:,idx1))+0.5 min(X_train(:,idx2))-0.5 max(X_train(:,idx2))+0.5])
title('SVM applied on the training data')
%% Ploting the result on the testing data
id1_test=find(y_test==1);
id2_test=find(y_test==2);
idx1=3;
idx2=4;
id_pred_test1 = find(X_test(:,[idx1,idx2])*W([idx1,idx2])< 0);
id_pred_test2 = find(X_test(:,[idx1,idx2])*W([idx1,idx2])>= 0);
figure()
set(gcf,'color','w')
plot(X_test(id1_test,idx1),X_test(id1_test,idx2),'ro')
hold on
plot(X_test(id_pred_test1,idx1),X_test(id_pred_test1,idx2),'rx')
hold on
plot(X_test(id2_test,idx1),X_test(id2_test,idx2),'bo')
hold on
plot(X_test(id_pred_test2,idx1),X_test(id_pred_test2,idx2),'bx')
hold on
grid on
x=min(X_test(:,idx1))-0.5:0.1:max(X_test(:,idx1))+0.5;
plot(x,(-W(idx1)/W(idx2))*x,'g','linewidth',2)
legend('true versicolor', 'pred versicolor ','true virginica', 'pred virginica','SVM Boundary')
title('SVM applied on the testing data')
axis([min(X_test(:,idx1))-0.5 max(X_test(:,idx1))+0.5 min(X_test(:,idx2))-0.5 max(X_test(:,idx2))+0.5])
%% Accuracy Test (Out-of-Sample performance)
id_pred_test1 = find(X_test*W< 0);
id_pred_test2 = find(X_test*W>= 0);
y_test_predict=zeros(length(X_test),1);
y_test_predict(id_pred_test1)=1;
y_test_predict(id_pred_test2)=2;
accuracy_test=1-sum(abs(y_test-y_test_predict))/length(X_test);
fprintf('The accuracy of the testing data is %1.1f\n',100*accuracy_test);
%% Question3: Trying for different values of tau
tau=0.001:0.01:5;
accuracy_test=zeros(length(tau),1);
accuracy_train=zeros(length(tau),1);
x0_primal=[zeros(d,1);5.*ones(n,1)];
for i=1:length(tau)
    [Q_primal,p_primal,A_primal,b_primal] = transform_svm_primal(tau(i),X_train,y_train);
    mu=100;
    tol=0.001;
    [X_primal,~] = barr_method(Q_primal,p_primal,A_primal,b_primal,x0_primal,mu,tol);
    W=X_primal(1:4);
    id_pred_test1 = find(X_test*W<= 0);
    id_pred_test2 = find(X_test*W> 0);
    y_test_predict=zeros(length(X_test),1);
    y_test_predict(id_pred_test1)=1;
    y_test_predict(id_pred_test2)=2;
    accuracy_test(i)=1-sum(abs(y_test-y_test_predict))/length(X_test);
    id_pred_train1 = find(X_train*W<= 0);
    id_pred_train2 = find(X_train*W> 0);
    y_train_predict=zeros(length(X_train),1);
    y_train_predict(id_pred_train1)=1;
    y_train_predict(id_pred_train2)=2;
    accuracy_train(i)=1-sum(abs(y_train-y_train_predict))/length(X_train);
end 
figure(100)
set(gcf,'color','w')
plot(tau,accuracy_train,'r','linewidth',2)
hold on
grid on
plot(tau,accuracy_test,'b','linewidth',2)
xlabel('\tau')
ylabel('Accuracy')
legend('accuracy train','accuracy test')
%% Question4a Plot the duality gap
tau=0.001;
[Q,p,A,b] = transform_svm_primal(tau,X_train,y_train);
[n,d]=size(X_train);
x0_primal=[zeros(d,1);2.*ones(n,1)];
color='gbry';
mu=[2,15,50,100];
tol=0.000001;
newton_iter=[];
for i=1:4
    [X_primal,xhist,Gap1,loss] = barr_method(Q,p,A,b,x0_primal,mu(i),tol);
    iter=length(Gap1);
    newton_iter=[newton_iter iter];
    set(gcf,'color','w')
    plot(Gap1,'linewidth',2,'color',color(i))
    ylabel('duality Gap')
    xlabel('iterations')
    hold on 
end
legend('\mu =2','\mu=15','\mu=50','\mu=100')
%% Question4b: Plot the number of iteration w.r.t.\mu
figure()
set(gcf,'color','w')
plot(mu,newton_iter,'r','linewidth',2)
grid on
ylabel('Barrier Iterations')
xlabel('\mu')
title('Barrier iterations w.r.t \mu')
hold on 
%% Question 4c semilogx of the primal (chaging mu)
tau = 0.001;
[n,d] = size(X_train);
[Q,p,A,b] = transform_svm_primal(tau,X_train,y_train); % Transform to a QP problem
x0_primal=[zeros(d,1);4.*ones(n,1)]; % Initialize with a stricly feasible point
tol = 10^-7;
mu = [2 15 50 100];
nb_iter = [];
for i=1:length(mu)
    [~,~,Gap,loss] = barr_method(Q,p,A,b,x0_primal,mu(i),tol);
    nb_iter = [nb_iter length(loss)];
    figure(20)
    set(gcf,'color','w')
    subplot(length(mu),1,i)
    semilogx(loss,'-*r','linewidth',2)
    grid on,
    xlabel('Barrier iterations');
    ylabel('Objective');
    xlim([1 length(loss)])
    title(sprintf('Objective primal function values\n(\\tau=%.3f, \\mu=%.2f)',tau,mu(i)))
    figure(21)
    set(gcf,'color','w')
    subplot(length(mu),1,i)
    semilogx(Gap,'-*b','linewidth',2)
    grid on,
    xlabel('Barrier iterations');
    ylabel('Dual Gap');
    xlim([1 length(loss)])
    title(sprintf('Duality Gap values\n(\\tau =%.3f, \\mu=%.2f)',tau,mu(i)))
end
%% Question 4d :semilogx of the dual (changing mu)
tau = 0.001;
[n,d] = size(X_train);
[Q,p,A,b] = transform_svm_dual(tau,X_train,y_train); % Transform to a QP problem
x0_dual=(1/(2*tau*n)).*ones(n,1);% Initialize with a stricly feasible point
tol = 10^-7;
mu = [2 15 50 100];
nb_iter = [];

for i=1:length(mu)
    [~,~,Gap,loss] = barr_method(Q,p,A,b,x0_dual,mu(i),tol);
    nb_iter = [nb_iter length(loss)];
    figure(22)
    set(gcf,'color','w')
    subplot(length(mu),1,i)
    semilogx(-loss,'-*r','linewidth',2)
    grid on,
    xlabel('Barrier iterations');
    ylabel('Objective');
    xlim([1 length(loss)])
    title(sprintf('Objective dual function values\n(\\tau=%.3f, \\mu=%.2f)',tau,mu(i)))
    figure(23)
    set(gcf,'color','w')
    subplot(length(mu),1,i)
    semilogx(Gap,'-*b','linewidth',2)
    grid on,
    xlabel('Barrier iterations');
    ylabel('dual Gap');
    xlim([1 length(loss)])
    title(sprintf('Duality gap values\n(\\tau=%.3f, \\mu=%.2f)',tau,mu(i)))
end
