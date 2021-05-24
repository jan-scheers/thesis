N = 16;
xin = linspace(0,1,N);
y = -sin(.8*pi*xin);

w1 = sdpvar(8,1);
b1 = sdpvar(8,1);
x1 = sdpvar(8,N);
w2 = sdpvar(8,8,'full');
b2 = sdpvar(8,1);
x2 = sdpvar(8,N);
w3 = sdpvar(1,8);
b3 = sdpvar(1,1);


assign(w1,2*rand(8,1)-1);
assign(b1,2*rand(8,1)-1);
assign(x1,tansig(value(w1*xin+repmat(b1,1,N))));
assign(w2,2*rand(8,8)-1);
assign(b2,2*rand(8,1)-1);
assign(x2,tansig(value(w2*x1 +repmat(b2,1,N))));
assign(w3,2*rand(1,8)-1);
assign(b3,2*rand(1,1)-1);

res = w3*x2 + b3 - y;
obj = res*res';

f1 = (x1-(w1*xin+repmat(b1,1,N)));
f2 = (x2-(w2*x1 +repmat(b2,1,N)));
con = [f1 >= 0; x1 >= 0; f1.*x1 <= 0;
      f2 >= 0; x2 >= 0; f2.*x2 <= 0];
% con = [x1 == max(w1*xin+repmat(b1,1,N),0);        x2 == max(w2*x1 +repmat(b2,1,N),0)];
% con = [x1 == tansig(w1*xin+repmat(b1,1,N));
%        x2 == tansig(w2*x1 +repmat(b2,1,N))];
ops = sdpsettings('usex0',1,'solver','fmincon');
ops.fmincon.MaxFunEvals = 20000;
ops.fmincon.MaxIter = 40;
ops.fmincon.PlotFcn = 'optimplotfval';

optimize(con,obj,ops);

%%
xs = linspace(0,1,1000);
x1s = poslin(value(w1)*xs+value(b1));
x2s = poslin(value(w2)*x1s+value(b2));
ys = value(w3)*x2s+value(b3);

hold off
plot(xin,y,'Linewidth',3);
hold on
plot(xs,ys,'g--','Linewidth',4);


