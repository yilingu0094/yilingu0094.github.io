DeepBlue  = [0,92/255,175/255];
DeepGreen = [92/255,172/255,129/255];
MyOrange  = [226/255,148/255,59/255];
MyBlue    = [123/255,144/255,210/255];
MyRed     = [224/255,60/255,138/255];

x=linspace(-1,1,50);
y=linspace(-1,1,50);
[X,Y] = meshgrid(x,y);
Z = X.^2-Y.^2;
figure;
surf(X,Y,Z,"EdgeColor","flat","FaceAlpha",1);
view(30,30);
colorbar('AxisLocation','out','Ticks',[-.8,.8],...
         'TickLabels',{'Low','High'})
xticks('');
yticks('');
zticks('');
zlim([-1,1]);
axis off;

plot(x,x.^2,'LineWidth',4,'Color',MyBlue);
axis off;
set(gcf,'Renderer', 'painters');


