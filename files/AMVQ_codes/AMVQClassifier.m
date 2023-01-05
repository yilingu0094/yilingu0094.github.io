classdef AMVQClassifier < handle


    properties
        prototype_list;
        label_list;
        radius_list;
    end


    methods
    function obj = AMVQClassifier()
        obj.prototype_list=[];
        obj.label_list=[];
        obj.radius_list=[];
     end
    function y = huber_norm(obj,x,mu)
        if norm(x,2)<=mu
            y = (1/(2*mu))*(norm(x,2)^2);
        else
            y = norm(x,2)-(1/2)*mu;
        end
    end
    
    function y = huber_gradient(obj,x,mu)
        if norm(x,2)<=mu
            y = x/mu;
        else
            y = x/norm(x,2);
        end
    end
    
    function y = hinge_func(obj,m,gamma)
        if m<=0
            y = 0;
        elseif m>=gamma
            y= 1;
            %y =100*(m-gamma/2);
        else
            y = (m^2)/(2*gamma);
        end
    end
    
    function y = hinge_gradient(obj,m,gamma)
        if m<=0
            y = 0;
        elseif m>=gamma
            y = 1;
        else
            y = m/gamma;
        end
    end
    
    function y = loss_func(obj,X,A,lambda,r,X_other)
        sum = 0;
        mu=0.01;
        gamma=0.01;
        for i = 1:size(X,1)
            sum = sum + hinge_func(obj,norm(X(i,:)-A,2)-r,gamma);
        end
        y = sum+lambda*r^2;
    end
    
    function [A,radius] = compute_loss(obj,X,X_other)
        cur=0;
        n = size(X,1); %行数
        m = size(X,2); %列数
        A = X(1,:); %设置初始点
    
        while cur<1000 %最多找1000次
            radius = min_radius(obj,A,X_other);
            lamda = 100;
            options = optimoptions(@fminunc,'Algorithm','quasi-newton');%拟牛顿参数
            pre_f=loss_func(obj,X,A,lamda,radius,X_other); %旧函数值
            A = fminunc(@(A)loss_func(obj,X,A,lamda,radius,X_other),A,options); %优化求新的A
            cur_f=loss_func(obj,X,A,lamda,radius,X_other); %新函数值
            if pre_f-cur_f < 0.1 %跳出循环条件
                break;
            end
            cur=cur+1;
        end
    end
    
    function [prototype_list,radius_list] = loop_func(obj,X,X_other,tol,outlier_coef)
        num = size(X,1); %总行数
        temp_X = X; %临时样本矩阵
        temp_prototype_list=[]; %保存prototype
        temp_radius_list=[]; %保存radius
        while size(temp_X,1)/num > tol
            count = 0;
            [A,radius] = compute_loss(obj,temp_X,X_other); %计算出本轮的prototype和r
            prototype = A;
            new_X = []; 
            for i = 1:size(temp_X,1) %删除小于半径样本
                if norm(prototype-temp_X(i,:),2) > radius
                    new_X = [new_X;temp_X(i,:)]; 
                else
                    count = count+1;
                end
            end
            temp_X = new_X; %更新temp_X
            if count/num < outlier_coef
                continue;
            end
            disp('************************')
            disp(prototype);
            disp('************************')
            temp_prototype_list=[temp_prototype_list;prototype]; %加入到prototype_list
            temp_radius_list=[temp_radius_list;radius];
        end
        prototype_list=temp_prototype_list;
        radius_list=temp_radius_list;
    end
   
    
    function y = min_radius(obj,A,X_other)%找最小半径
        n = size(X_other,1);
        min = 10^10;
        for i =1:n
            if min>norm(A-X_other(i,:),2)
                min = norm(A-X_other(i,:),2);
           
            end
        end
        y = min/2;
    end
    function y = train(obj,X_train,Y_train)
        tic
        %[X,X_other]=twospirals;
        %[X,X_other]=halfkernel;
        M = containers.Map('KeyType','uint64','ValueType','any'); %创建一个空的MAP
        %keys={};
        %sample=[];
        num=size(X_train,1);%数量
        for i=1:num
            if isKey(M,Y_train(i)) %存在key
                temp_list=M(Y_train(i));
                temp_list=[temp_list;X_train(i,:)];
                M(Y_train(i))=temp_list;
            else %不存在key
                new_list=[X_train(i,:)];
                M(Y_train(i))=new_list;
            end
        end

        tol=0.2;
        outlier_coef=0.001;
        keySet = keys(M);
        disp(keySet)
        %disp(M);
        %disp(length(keySet));
        for i=1:length(keySet)
            X=M(keySet{i});
            X_other=[];
            for j=1:length(keySet)
                if j ~= i
                    X_other=[X_other;M(keySet{j})];
                end
            end
            
            [prototype_list,radius_list]=loop_func(obj,X,X_other,tol,outlier_coef);%入口
            obj.prototype_list=[obj.prototype_list;prototype_list];
            obj.radius_list=[obj.radius_list;radius_list];
            for j=1:size(prototype_list,1)
                obj.label_list=[obj.label_list;keySet{i}]
            end
        end
        % hold on;
        %for i=1:size(prototype_list,1)
        %plot(prototype_list(i,1),prototype_list(i,2),"Marker","+");
        %x = prototype_list(i,1);
        % y = prototype_list(i,2);
        % r = radius_list(i);
        %rectangle('Position', [x-r,y-r,2*r,2*r], 'Curvature', [1 1],'EdgeColor', 'r');
        %end
        %hold off;
        %disp('prototype为');
        %disp(prototype_list);
        %disp('半径为')
        %disp(radius_list);
        toc
        disp(['运行时间: ',num2str(toc)])
    end

    function y_test = predict(obj,X_test)
        n=size(X_test,1);
        y_test=[];
       % disp(keySet);
        for i=1:n
            %temp_dis=obj.prototypeMap(keySet{1});
           % disp(keySet{1});
           % disp(X_test(i,:));
            min=norm(X_test(i,:)-obj.prototype_list(1),2)-obj.radius_list(1);
            y_test(i)=obj.label_list(1);
            %keySet = keys(obj.prototypeMap);
            for j=2:size(obj.prototype_list,1)
                %temp_list=obj.prototypeMap(keySet{j});
                dis=norm(X_test(i,:)-obj.prototype_list(j),2)-obj.radius_list(j);
                if dis<min
                    min=dis;
                    y_test(i)=obj.label_list(j);
                end
            end
        end
    end
    end
end