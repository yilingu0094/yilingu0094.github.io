clear;
DeepBlue  = [0,92/255,175/255];
DeepGreen = [92/255,172/255,129/255];
MyOrange  = [226/255,148/255,59/255];
MyBlue    = [123/255,144/255,210/255];
MyRed     = [224/255,60/255,138/255];

x = linspace(0,10,50)';
f=@(x) 0.2.*x + 1;
y = f(x)-1+2*rand(length(x),1);
figure;
scatter(x,y,'filled',"MarkerFaceColor",MyOrange);
hold on;
plot(-1:11,f(-1:11),"LineStyle","-","LineWidth",2,"Color",MyBlue);
xlim([-1,11]);
ylim([-1,5]);
axis off;
