function st = bicycle_step(cfg, st, deltaCmd, v)
%BICYCLE_STEP Kinematic bicycle (rear-axle reference) + steering actuator.
%   st = [x y psi delta].  delta' = sat((delta_cmd - delta)/tau, rate)
%   x' = v cos(psi),  y' = v sin(psi),  psi' = v tan(delta) / L
%   (Kong et al., IEEE IV 2015). Semi-implicit Euler, dt = cfg.dt.
dt = cfg.dt;
deltaCmd = min(max(deltaCmd, -cfg.maxSteer), cfg.maxSteer);
rate = (deltaCmd - st(4)) / cfg.steerTau;
rate = min(max(rate, -cfg.steerRate), cfg.steerRate);
d = min(max(st(4) + rate*dt, -cfg.maxSteer), cfg.maxSteer);
x = st(1) + v*cos(st(3))*dt;
y = st(2) + v*sin(st(3))*dt;
psi = st(3) + v/cfg.L*tan(d)*dt;
psi = atan2(sin(psi), cos(psi));
st = [x y psi d];
end
