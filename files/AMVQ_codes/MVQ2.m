function result = MVQ2
    result.huber_norm=@huber_norm;
    result.huber_gradient=@huber_gradient;
    result.hinge_func=@hinge_func;
    result.hinge_gradient=@hinge_gradient;
    result.loss_func=@loss_func;
    result.radius_func=@radius_func;
    result.compute_loss=@compute_loss;
    result.loop_func=@loop_func;
    result.min_radius=@min_radius;
    result.GeneRand_Category=@GeneRand_Category;
    result.DataMaker1=@DataMaker1;
    result.main_func=@main_func;
    result.DataMaker2=@DataMaker2;
    result.twospirals=@twospirals;
    result.clusterincluster=@clusterincluster;
    result.crescentfullmoon=@crescentfullmoon;
    result.halfkernel=@halfkernel;
    result.corners=@corners;
    result.outlier=@outlier;
    result.prediction=@prediction;
end


function y = huber_norm(x,mu)
    if norm(x,2)<=mu
        y = (1/(2*mu))*(norm(x,2)^2);
    else
        y = norm(x,2)-(1/2)*mu;
    end
end

function y = huber_gradient(x,mu)
    if norm(x,2)<=mu
        y = x/mu;
    else
        y = x/norm(x,2);
    end
end

function y = hinge_func(m)
    if m<=0
        y = 0;
    elseif m>0
        y= 1;
    end
end

function y = hinge_gradient(~)
    y=0;
end

function y = loss_func(X,A,lambda,r,X_other)
    sum = 0;
    mu=0.01;
    for i = 1:size(X,1)
        sum = sum + hinge_func(norm(X(i,:)-A,2)-r);
    end
    y = sum+lambda*r^2;
end

function [A,radius] = compute_loss(X,X_other)
    cur=0;
    n = size(X,1); %行数
    m = size(X,2); %列数
    A = X(1,:); %设置初始点

    while cur<1000 %最多找1000次
        radius = min_radius(A,X_other);
        lamda = 100;
        options = optimoptions(@fminunc,'Algorithm','quasi-newton');%拟牛顿参数
        pre_f=loss_func(X,A,lamda,radius,X_other); %旧函数值
        A = fminunc(@(A)loss_func(X,A,lamda,radius,X_other),A,options); %优化求新的A
        cur_f=loss_func(X,A,lamda,radius,X_other); %新函数值
        if pre_f-cur_f < 0.1 %跳出循环条件
            break;
        end
        cur=cur+1;
    end
end

function [prototype_list,radius_list] = loop_func(X,X_other,tol,outlier_coef)
    num = size(X,1); %总行数
    temp_X = X; %临时样本矩阵
    temp_prototype_list=[]; %保存prototype
    temp_radius_list=[]; %保存radius
    while size(temp_X,1)/num > tol
        count = 0;
        [A,radius] = compute_loss(temp_X,X_other); %计算出本轮的prototype和r
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
        temp_prototype_list=[temp_prototype_list;prototype]; %加入到prototype_list
        temp_radius_list=[temp_radius_list;radius];
    end
    prototype_list=temp_prototype_list;
    radius_list=temp_radius_list;
end


function y = min_radius(A,X_other)%找最小半径
    n = size(X_other,1);
    min = 10^10;
    for i =1:n
        if min>norm(A-X_other(i,:),2)
            min = norm(A-X_other(i,:),2);
        end
    end
    y = min/2;
end






function y = prediction(x_pred)%预测某一个向量x_pred的label
    [prototype_list,~] = loop_func(X,X_other,tol,outlier_coef);
    dist = {};
    valueset = [];
    for i = prototype_list
        valueset = [valueset;i];
        dist = {dist;norm(i-x_pred,2)};
    end
    M = containers.Map(dist,valueset);
    min_prototype = M(min(vertcat(dist{[1 2]})));
end




    






function [X,X_other] = DataMaker1()
    X=normrnd(2,1,1000,2); %高斯分布的第一类数据集合
    X_other=normrnd(6,1,500,2);%第二类
    X_other=[X_other;normrnd(-2,1,500,2)];
    figure;
    hold on;
    plot(X(1:1000,1),X(1:1000,2),'.');
    plot(X_other(1:1000,1),X_other(1:1000,2),'.');
    hold off;
end

function [X,X_other] =DataMaker2()
    X=[];
    for i = 1:1000
        X(i,1)=unifrnd(2,3);
        X(i,2)=unifrnd(2,8);
    end

    for i =1001:2000
        X(i,1)=unifrnd(2,9);
        X(i,2)=unifrnd(8,9);
    end

    for i =2001:3000
        X(i,1)=unifrnd(8,9);
        X(i,2)=unifrnd(2,8);
    end

    X_other=normrnd(5,0.6,1000,2);
    figure;
    hold on;
    plot(X(1:3000,1),X(1:3000,2),'.');
    plot(X_other(1:1000,1),X_other(1:1000,2),'.');
    hold off;
end

function [X,X_other]=DataMaker3()
        X=[];
    for i = 1:500
        X(i,1)=unifrnd(2,3);
        X(i,2)=unifrnd(2,8);
    end

    for i =501:1000
        X(i,1)=unifrnd(2,9);
        X(i,2)=unifrnd(8,9);
    end

    for i =1001:1500
        X(i,1)=unifrnd(8,9);
        X(i,2)=unifrnd(2,8);
    end


        X_other=[];
    for i = 1:500
        X_other(i,1)=unifrnd(5,6);
        X_other(i,2)=unifrnd(2,7);
    end

    for i =501:1000
        X_other(i,1)=unifrnd(5,11);
        X_other(i,2)=unifrnd(1,2);
    end

    for i =1001:1500
        X_other(i,1)=unifrnd(10,11);
        X_other(i,2)=unifrnd(2,7);
    end

    figure;
    hold on;
    plot(X(1:1500,1),X(1:1500,2),'.');
    plot(X_other(1:1500,1),X_other(1:1500,2),'.');
    hold off;
end

function y = main_func()
    tic
    %[X,X_other]=twospirals;
    [X,X_other]=halfkernel;
    %[X,X_other]=crescentfullmoon;
    %[X,X_other]=outlier;
    %tol=0.01;
    %outlier_coef=0.001;
    %tol=0.05;
    %outlier_coef=0.005;
    tol=0.1;
    outlier_coef=0.01;
    [prototype_list,radius_list]=loop_func(X_other,X,tol,outlier_coef);%入口
    hold on;
    for i=1:size(prototype_list,1)
    %plot(prototype_list(i,1),prototype_list(i,2),"Marker","+");
    x = prototype_list(i,1);
    y = prototype_list(i,2);
    r = radius_list(i);
    rectangle('Position', [x-r,y-r,2*r,2*r], 'Curvature', [1 1],'EdgeColor', 'r');
    end
    hold off;
    disp('prototype: ');
    disp(prototype_list);
    disp('radius:')
    disp(radius_list);
    toc
    disp(['time: ',num2str(toc)])

end

function y = GeneRand_Category()
    %   此处显示详细说明
    Category1=[normrnd(2,1,100,2)]; %高斯分布的第一类数据集合
    Category2=[normrnd(10,1,100,2)];%第二类
    Category3=[normrnd(2,1,100,2)];%第三类
    for i=1:100
        Category3(i,1)= Category3(i,1)+8;
    end
    figure;
    hold on;
    plot(Category1(1:100,1),Category1(1:100,2),'.');
    plot(Category2(1:100,1),Category2(1:100,2),'.');
    plot(Category3(1:100,1),Category3(1:100,2),'.');
    hold off;
end

function [X,X_other] = twospirals(N, degrees, start, noise)
% Generate "two spirals" dataset with N instances.
% degrees controls the length of the spirals
% start determines how far from the origin the spirals start, in degrees
% noise displaces the instances from the spiral. 
%  0 is no noise, at 1 the spirals will start overlapping

    if nargin < 1
        N = 2000;
    end
    if nargin < 2
        degrees = 570;
    end
    if nargin < 3
        start = 90;
    end
    if nargin < 5
        noise = 0.2;
    end  
    
    deg2rad = (2*pi)/360;
    start = start * deg2rad;

    N1 = floor(N/2);
    N2 = N-N1;
    
    n = start + sqrt(rand(N1,1)) * degrees * deg2rad;   
    X = [-cos(n).*n + rand(N1,1)*noise sin(n).*n+rand(N1,1)*noise zeros(N1,1)];
    
    n = start + sqrt(rand(N1,1)) * degrees * deg2rad;      
    X_other = [cos(n).*n+rand(N2,1)*noise -sin(n).*n+rand(N2,1)*noise ones(N2,1)];
    

    figure;
    hold on;
    plot(X(1:1000,1),X(1:1000,2),'.');
    plot(X_other(1:1000,1),X_other(1:1000,2),'.');
    hold off;
end


function [X,X_other] = crescentfullmoon(N, r1, r2, r3)

    if nargin < 1
        N = 2000;
    end
    if mod(N,4) ~= 0
        N = round(N/4) * 4;
    end
    if nargin < 2
        r1 = 5;
    end
    if nargin < 3
        r2 = 10;
    end
    if nargin < 4
        r3 = 15;
    end
    
    N1 = N/4;
    N2 = N-N1;
    
    phi1 = rand(N1,1) * 2 * pi;
    R1 = sqrt(rand(N1, 1));
    X = [cos(phi1) .* R1 * r1 sin(phi1) .* R1 * r1 zeros(N1,1)];
    
    d = r3 - r2;
    phi2 = pi + rand(N2,1) * pi;
    R2 = sqrt(rand(N2,1));
    X_other = [cos(phi2) .* (r2 + R2 * d) sin(phi2) .* (r2 + R2 * d) ones(N2,1)];

    figure;
    hold on;
    plot(X(1:500,1),X(1:500,2),'.');
    plot(X_other(1:500,1),X_other(1:500,2),'.');
    hold off;
end


function [X,X_other] = halfkernel(N, minx, r1, r2, noise, ratio)
    
    if nargin < 1
        N = 1000;
    end
    if mod(N,2) ~= 0
        N = N + 1;
    end
    if nargin < 2
        minx = -20;
    end
    if nargin < 3
        r1 = 20;
    end
    if nargin < 4
        r2 = 35;
    end
    if nargin < 5
        noise = 4;
    end
    if nargin < 6
        ratio = 0.6;
    end
    
    phi1 = rand(N/2,1) * pi;
    X = [minx + r1 * sin(phi1) - .5 * noise  + noise * rand(N/2,1) r1 * ratio * cos(phi1) - .5 * noise + noise * rand(N/2,1) ones(N/2,1)];
        
    phi2 = rand(N/2,1) * pi;
    X_other = [minx + r2 * sin(phi2) - .5 * noise  + noise * rand(N/2,1) r2 * ratio * cos(phi2) - .5 * noise  + noise * rand(N/2,1) zeros(N/2,1)];

    figure;
    hold on;
    plot(X(1:500,1),X(1:500,2),'.');
    plot(X_other(1:500,1),X_other(1:500,2),'.');
    hold off;
end


function [X,X_other] = outlier(N, r, dist, outliers, noise)

    if nargin < 1
        N = 2000;
    end
    if nargin < 2
        r = 20;
    end
    if nargin < 3
        dist = 30;
    end
    if nargin < 4
        outliers = 0.04;
    end
    if nargin < 5
        noise = 5;
    end

    N1 = round(N * (.5-outliers));
    N2 = N1;
    N3 = round(N * outliers);
    N4 = N-N1-N2-N3;

    phi1 = rand(N1,1) * pi;
    r1 = sqrt(rand(N1,1))*r;
    X = [-dist + r1.*sin(phi1) r1.*cos(phi1) zeros(N1,1)];

    phi2 = rand(N2,1) * pi;
    r2 = sqrt(rand(N2,1))*r;
    X_other = [dist - r2.*sin(phi2) r2.*cos(phi2) 3*ones(N2,1)];    
    
    figure;
    hold on;
    plot(X(1:500,1),X(1:500,2),'.');
    plot(X_other(1:500,1),X_other(1:500,2),'.');
    hold off;

end

function data = corners(N, scale, gapwidth, cornerwidth)

    if nargin < 1
        N = 1000;
    end
    if mod(N,8) ~= 0
        N = round(N/8) * 8;
    end

    if nargin < 2
        scale = 10;
    end
    if nargin < 3
        gapwidth = 2;
    end   
    if nargin < 4
        cornerwidth = 2;
    end

    perCorner = N/4;

    xplusmin = [ones(perCorner,1); -1*ones(perCorner,1); ones(perCorner,1); -1*ones(perCorner,1)];
    yplusmin = [ones(perCorner,1); -1*ones(2*perCorner,1); ones(perCorner,1)];
    
    horizontal = [xplusmin(1:2:end) * gapwidth + xplusmin(1:2:end) * scale .* rand(N/2,1), ...
                  yplusmin(1:2:end) * gapwidth + cornerwidth * yplusmin(1:2:end) .* rand(N/2,1), ...
                  floor((0:N/2-1)'/(perCorner*.5))];
       
    vertical = [xplusmin(2:2:end) * gapwidth + cornerwidth * xplusmin(2:2:end) .* rand(N/2,1), ...
                yplusmin(2:2:end) * gapwidth + yplusmin(2:2:end) * scale .* rand(N/2,1), ...
                floor((0:N/2-1)'/(perCorner*.5))];
    
    data=  [horizontal; vertical];

    figure;
    hold on;
    plot(data(1:1000,1),data(1:1000,2),'.');
    hold off;
end
