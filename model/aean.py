import numpy as np

class AEANConfig:
   def __init__(self, input_dim, output_dim, learning_rate=0.01, 
                primary_update_ratio=10, use_bias=False, 
                weight_init='zeros', dtype=np.float32):
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.learning_rate = learning_rate
       self.primary_update_ratio = primary_update_ratio
       self.use_bias = use_bias
       self.weight_init = weight_init
       self.dtype = dtype

class AdaptiveErrorAbsorptionNetwork:
   def __init__(self, config):
       self.config = config
       self.input_dim = config.input_dim
       self.output_dim = config.output_dim
       self.lr = config.learning_rate
       self.K = config.primary_update_ratio
       self.dtype = config.dtype
       if config.weight_init == 'zeros':
           self.weights = np.zeros((config.output_dim, config.input_dim), dtype=config.dtype)
       elif config.weight_init == 'small':
           self.weights = np.random.randn(config.output_dim, config.input_dim).astype(config.dtype) * 0.01
       self.use_bias = config.use_bias
       if self.use_bias:
           self.bias = np.zeros(config.output_dim, dtype=self.dtype)
       else:
           self.bias = None
       self.step_counter = 0
       self.weight_history = []
       self.output_history = []
       self.error_history = []
       self.delta_history = []
   
   def forward(self, x):
       single_input = False
       if x.ndim == 1:
           x = x.reshape(1, -1)
           single_input = True
       y_aux = x @ self.weights.T
       if self.use_bias:
           y_aux += self.bias
       self.output_history.append(y_aux.copy())
       return y_aux.flatten() if single_input else y_aux
   
   def update(self, x, delta, primary_update_flag=False, primary_weights=None, primary_update_fn=None):
       if x.ndim > 1:
           x = x.flatten()
       if isinstance(delta, (int, float)):
           delta = np.array([delta], dtype=self.dtype)
       if delta.ndim == 0:
           delta = delta.reshape(1)
       
       self.delta_history.append(delta.copy())
       
       gradient = np.outer(delta, x)
       weight_update = self.lr * gradient
       self.weights -= weight_update
       
       bias_update = None
       if self.use_bias:
           bias_gradient = delta
           bias_update = self.lr * bias_gradient
           self.bias -= bias_update
       
       primary_updated = False
       self.step_counter += 1
       
       if primary_weights is not None and primary_update_fn is not None:
           should_update = primary_update_flag or (self.step_counter % self.K == 0)
           if should_update:
               primary_update_fn(primary_weights, delta, x)
               primary_updated = True
               self.step_counter = 0
       
       self.weight_history.append(self.weights.copy())
       
       update_info = {
           'aux_weight_update_norm': np.linalg.norm(weight_update),
           'aux_weight_norm': np.linalg.norm(self.weights),
           'gradient_norm': np.linalg.norm(gradient),
           'primary_updated': primary_updated,
           'step': self.step_counter,
           'delta_norm': np.linalg.norm(delta)
       }
       
       if self.use_bias:
           update_info['bias_update_norm'] = np.linalg.norm(bias_update)
           update_info['bias_norm'] = np.linalg.norm(self.bias)
       
       self.error_history.append({
           'delta': delta.copy(),
           'aux_output': self.output_history[-1] if self.output_history else None,
           'weight_norm': np.linalg.norm(self.weights)
       })
       
       return update_info
   
   def fuse_outputs(self, y_main, x=None, update_aux=True):
       if update_aux:
           assert x is not None
           y_aux = self.forward(x)
       else:
           y_aux = self.output_history[-1] if self.output_history else np.zeros_like(y_main)
       
       if isinstance(y_main, (int, float)):
           y_main = np.array([y_main])
       if isinstance(y_aux, (int, float)):
           y_aux = np.array([y_aux])
       
       return y_main + y_aux
   
   def get_correction_signal(self, psnr, acc, psnr_target=35.0, acc_target=0.95):
       delta_psnr = psnr_target - psnr
       delta_acc = acc_target - acc
       delta = 0.3 * delta_psnr + 0.7 * delta_acc
       return delta
   
   def reset(self, keep_weights=False):
       self.step_counter = 0
       self.output_history = []
       self.error_history = []
       self.delta_history = []
       
       if not keep_weights:
           if self.config.weight_init == 'zeros':
               self.weights = np.zeros((self.output_dim, self.input_dim), dtype=self.dtype)
           elif self.config.weight_init == 'small':
               self.weights = np.random.randn(self.output_dim, self.input_dim).astype(self.dtype) * 0.01
           if self.use_bias:
               self.bias = np.zeros(self.output_dim, dtype=self.dtype)
           self.weight_history = []
       else:
           self.weight_history = [self.weights.copy()]
   
   def get_convergence_status(self, window_size=50, threshold=1e-4):
       if len(self.weight_history) < window_size + 1:
           return False
       recent_weights = np.array(self.weight_history[-window_size:])
       weight_changes = np.diff(recent_weights, axis=0)
       mean_abs_change = np.mean(np.abs(weight_changes))
       return mean_abs_change < threshold
   
   def get_error_statistics(self):
       if len(self.delta_history) == 0:
           return {'mean_delta': 0, 'std_delta': 0}
       deltas = np.concatenate([d.flatten() for d in self.delta_history])
       return {
           'mean_delta': np.mean(deltas),
           'std_delta': np.std(deltas),
           'max_delta': np.max(np.abs(deltas))
       }