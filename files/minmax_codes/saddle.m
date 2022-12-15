x=linspace(-1,1,50);
y=linspace(-1,1,50);
[X,Y] = meshgrid(x,y);
Z = X.^2-Y.^2;
figure;
surf(X,Y,Z,"EdgeColor","flat","FaceAlpha",1);
view(30,30);

hold on;
scatter3(0,0,0.05,'filled','k');
colorbar('AxisLocation','out','Ticks',[-.8,.8],...
         'TickLabels',{'Low','High'})
xticks('');
yticks('');
zticks('');
zlim([-1,1]);
axis off;
set(gcf,'Renderer', 'painters');


