%% Parameter
T = [0.19, -0.92; -0.42, -0.28];
U = [0.61; -1.5];
V = [1.5, -0.81, -0.24];
W = [-1.4, -0.81; -2.2, -1.7; -0.27, -0.73];

x_init = [-1 1];
target = [0 1];
%% FP
z_r = x_init * T;
r = ReLU(x_init * T);

z_q = r * U;
q = Tanh(r * U);

z_p = q * V;
p = sigmoid(q * V);

z_y = p * W;
y = sigmoid(p * W);

%% BP Neuron Unit
dE_dy = Loss_dev(y, target);

dE_dp = dE_dy.* sigmoid_dev(z_y) * W';

dE_dq = dE_dp.* sigmoid_dev(z_p) * V';

dE_dr = dE_dq.* Tanh_dev(z_q) * U';

%% BP Weight Matrix
dE_dw = (dE_dy.* sigmoid_dev(z_y))' * p;

dE_dv = (dE_dp.* sigmoid_dev(z_p))' * q;

dE_du = (dE_dq.* Tanh_dev(z_q))' * r;

dE_dt = (dE_dr.* ReLU_dev(z_r))' * x_init;
