AMVQ=AMVQClassifier();
%X=normrnd(2,1,1000,2); %高斯分布的第一类数据集合
%X=[X;normrnd(6,1,1000,2)];%第二类
%Y=[];
%for i=1:1000
%    Y(i)=1;
%end
%for i=1001:2000
%    Y(i)=2;
%end
%disp(X);
%disp(Y);
X_train=xlsread('C:\Users\Administrator\Desktop\X_train.xlsx');
Y_train=xlsread('C:\Users\Administrator\Desktop\y_train.xlsx');
X_test=xlsread('C:\Users\Administrator\Desktop\X_test.xlsx');
Y_test=xlsread('C:\Users\Administrator\Desktop\y_test.xlsx');
Y_train = Y_train(:,1);
Y_test = Y_test(:,1);
disp(Y_train);
disp(Y_test);
AMVQ.train(X_train,Y_train);
%X_test=normrnd(2,1,100,2);
%X_test=[X_test;normrnd(6,1,100,2)];
y_test=AMVQ.predict(X_test);
disp(y_test);
count=0;
for i=1:size(X_test,1)
    if Y_test(i)==y_test(i)
        count=count+1;
    end
end
disp(count/size(X_test,1));
