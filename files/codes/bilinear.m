%% Trajectory of Sovling Bilinear Problem 
% min_x max_y f(x,y) = xy
% solution is (0,0)
% initial point (10,10)
% step size = 1
clear;

x0 = 10;
y0 = 10;
eta = .15;
maxit = 150;
x = zeros(maxit+1,1);
x(1) = x0;
y = x;

% Colors
DeepBlue  = [0,92/255,175/255];
DeepGreen = [92/255,172/255,129/255];
MyOrange  = [226/255,148/255,59/255];
MyBlue    = [123/255,144/255,210/255];
MyRed     = [224/255,60/255,138/255];
%% Proximal Point
for k = 1:maxit
    x(k+1) = (x(k) - eta * y(k))/(1+eta^2);
    y(k+1) = (y(k) + eta * x(k))/(1+eta^2);
end


plot(x,y,'LineWidth',2,'Color',DeepBlue,'LineStyle','-');

xlim([-15,15]);
ylim([-15,15]);
axis off;
hold on;

% quiver_idx = ceil(linspace(.01*maxit,.05*maxit,5));
% u = x(quiver_idx+3)-x(quiver_idx);
% v = y(quiver_idx+3)-y(quiver_idx);
% u = 2*u./sqrt(u.^2+v.^2);
% v = 2*v./sqrt(u.^2+v.^2);
% quiver(x(quiver_idx),y(quiver_idx),u,v,'off',"LineWidth",3,'Color',MyOrange);

%% OGDA
% for k = 1:maxit
%     if k == 1
%         x(k+1) = x(k) - eta * y(k);
%         y(k+1) = y(k) + eta * x(k);
%     else
%         x(k+1) = x(k) - 2*eta * y(k) + eta*y(k-1);
%         y(k+1) = y(k) + 2*eta * x(k) - eta*x(k-1);
%     end
% end
% plot(x,y,'LineWidth',2,'Color',MyOrange,'LineStyle','-.');
% 
% plot(0,0,'o','MarkerFaceColor',MyRed,'MarkerSize',6);
% plot(x0,y0,'o','MarkerFaceColor',MyOrange,'MarkerSize',8);
%legend('Proximal Point','Optimistic GDA');

%% Extra Gradient
for k = 1:maxit
    x_ = x(k) - eta * y(k);
    y_ = y(k) + eta * x(k);
    x(k+1) = x(k) - eta * y_;
    y(k+1) = y(k) + eta * x_;
end
plot(x,y,'LineWidth',2,'Color',MyOrange,'LineStyle','-.');

plot(0,0,'o','MarkerFaceColor',MyRed,'MarkerSize',6);
plot(x0,y0,'o','MarkerFaceColor',MyOrange,'MarkerSize',8);

%% Gradient Descent Ascent
% 
% for k = 1:maxit
%     x(k+1) = x(k) - eta * y(k);
%     y(k+1) = y(k) + eta * x(k);
% end
% 
% 
% plot(x,y,'LineWidth',2,'Color',MyBlue,'LineStyle','-');
% 
% xlim([-45,50]);
% ylim([-50,40]);
% axis off;
% 
% 
% hold on;
% 
% plot(0,0,'o','MarkerFaceColor',MyRed,'MarkerSize',8);
% plot(x0,y0,'o','MarkerFaceColor',MyOrange,'MarkerSize',8);
% %p = plot(x0,y0,'o','MarkerFaceColor',MyOrange,'MarkerSize',8);
% %axis square;
% %hold off;
% %xticks([]);
% %yticks([]);
% 
% quiver_idx = 5:20:maxit-1;
% u = x(quiver_idx+2)-x(quiver_idx);
% v = y(quiver_idx+2)-y(quiver_idx);
% u = 6*u./sqrt(u.^2+v.^2);
% v = 6*v./sqrt(u.^2+v.^2);
% quiver(x(quiver_idx),y(quiver_idx),u,v,'off',"LineWidth",3,'Color',MyOrange);
% 
% set(gcf,'Renderer', 'painters');
% % for k = 1:maxit
% %     p.XData = x(k);
% %     p.YData = y(k);
% %     pause(.01)
% %     drawnow
% %     frame = getframe(gca);
% %     filename = join(['GDA-',num2str(k),'.png']);
% %     imwrite(frame2im(frame),filename)
% % end

set(gcf,'Renderer', 'painters');
