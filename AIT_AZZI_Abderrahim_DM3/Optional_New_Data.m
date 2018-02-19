%% Optional Question: Use other data
Data_Train = load('classificationA.train');
Data_Test = load('classificationA.test');
X_train=Data_Train(:,1:2);
y_train=Data_Train(:,end);
X_train=X_train-mean(X_train); %Centering the data
X_test=Data_Test(:,1:2);
X_test=X_test-mean(X_test);
y_test=Data_Test(:,end); 
id1_train=find(y_train==0);
id2_train=find(y_train==1);
%% Plot the training and testing data
display_Data(Data_Train,Data_Test)
%% Aplying the barrier method
tau=0.001;
[Q,p,A,b] = transform_svm_primal(tau,X_train,y_train);
[n,d]=size(X_train);
x0=[zeros(d,1);2.*ones(n,1)];
mu=50;
tol=0.0000001;
[X_primal,~,~] = barr_method(Q,p,A,b,x0,mu,tol);
W=X_primal(1:2);
idx1=1;
idx2=2;
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
legend('true class 0', 'pred class 0 ','true class 1', 'pred class 1','SVM Boundary')
axis([min(X_train(:,idx1))-0.5 max(X_train(:,idx1))+0.5 min(X_train(:,idx2))-0.5 max(X_train(:,idx2))+0.5])
title('SVM applied on the training data')
%% Ploting the result on the testing data
id1_test=find(y_test==0);
id2_test=find(y_test==1);
idx1=1;
idx2=2;
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
%% Trying different values of tau
tau=0.001:0.1:2;
accuracy_test=zeros(length(tau),1);
accuracy_train=zeros(length(tau),1);
x0=[zeros(d,1);5.*ones(n,1)];
for i=1:length(tau)
    [Q,p,A,b] = transform_svm_primal(tau(i),X_train,y_train);
    mu=100;
    tol=0.00001;
    [X_primal,~] = barr_method(Q,p,A,b,x0,mu,tol);
    W=X_primal(1:2);
    id_pred_test1 = find(X_test*W<= 0);
    id_pred_test2 = find(X_test*W> 0);
    y_test_predict=zeros(length(X_test),1);
    y_test_predict(id_pred_test1)=0;
    y_test_predict(id_pred_test2)=1;
    accuracy_test(i)=1-sum(abs(y_test-y_test_predict))/length(X_test);
    id_pred_train1 = find(X_train*W<= 0);
    id_pred_train2 = find(X_train*W> 0);
    y_train_predict=zeros(length(X_train),1);
    y_train_predict(id_pred_train1)=0;
    y_train_predict(id_pred_train2)=1;
    accuracy_train(i)=1-sum(abs(y_train-y_train_predict))/length(X_train);
end 
figure(100)
set(gcf,'color','w')
plot(tau(2:end),accuracy_train(2:end),'r','linewidth',2)
hold on
plot(tau(2:end),accuracy_test(2:end),'b','linewidth',2)
xlabel('\tau')
grid on
ylabel('Accuracy')
legend('accuracy train','accuracy test')
title('accuracy w.r.t \tau')
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
    ylabel('Obj');
    xlim([1 length(loss)])
    title(sprintf('Objective dual function values\n(\\tau=%.3f, \\mu=%.2f)',tau,mu(i)))
    figure(23)
    set(gcf,'color','w')
    subplot(length(mu),1,i)
    semilogx(Gap,'-*b','linewidth',2)
    grid on,
    xlabel('Barrier iterations');
    ylabel('Obj');
    xlim([1 length(loss)])
    title(sprintf('Duality gap values\n(\\tau=%.3f, \\mu=%.2f)',tau,mu(i)))
end
