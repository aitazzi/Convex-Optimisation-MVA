function  display_Data( train,test)
%This function displays the cloud of data in 2D
ind_train0 = find(train(:,3)==0);
ind_train1 = find(train(:,3)==1);
ind_test0 = find(test(:,3)==0);
ind_test1 = find(test(:,3)==1);
N1 = length(ind_train0);
N2 = length(ind_train1);
data_train_1 = train(ind_train0,:);
data_train_2 = train(ind_train1,:);
data_test_1 = test(ind_test0,:);
data_test_2 = test(ind_test1,:);
set(gcf,'color','w')
set(gcf,'position',[100 200 750 450])
clf
subplot(1,2,1)
scatter(data_train_1(:,1),data_train_1(:,2),'ro');
hold on
grid on
scatter(data_train_2(:,1),data_train_2(:,2),'bo');
title('Scatter of the traning data')
legend('train class 0', 'train class 1')
axis([min(train(:,1))-0.5 max(train(:,1))+0.5 min(train(:,2))-0.5 max(train(:,2))+0.5])
%axis([-10 10 -8 8])
subplot(1,2,2)
scatter(data_test_1(:,1),data_test_1(:,2),'ro');
hold on
scatter(data_test_2(:,1),data_test_2(:,2),'bo');
title('Scatter of the testing data')
legend('test class 0', 'test class 1')
grid on
axis([min(test(:,1))-0.5 max(test(:,1))+0.5 min(test(:,2))-0.5 max(test(:,2))+0.5])
%axis([-10 10 -8 8])
end

