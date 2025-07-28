import torch 
import numpy as np

def sphere_trace_modified(origin, direction, sdf_fn, max_t, eps=1e-4, step_bound=10):
   
    all_pts = []
    all_pts_hit = []
    sdf_cost = 0
    
    p = origin + 0*direction
    d = sdf_fn(p)
    t = torch.zeros(d.shape).to(origin)
    
    iter = 0
    delta = torch.abs(d)
    not_converged = (t < max_t) # march every ray till the end 
    while iter < 2000 and torch.any(not_converged): 
        
        # add samples near surface and step over the surface for these samples 
        p_near_surface = (t < max_t) & (delta < eps)
        if torch.any(p_near_surface):
            p_hits = torch.ones_like(p).to(p)
            p_hits[p_near_surface] = p[p_near_surface]
            all_pts.append(p_hits)
            all_pts_hit.append(p_near_surface)
            
            # gather samples near surface 
            p_near = p[p_near_surface]
            delta_near = delta[p_near_surface]
            t_near = t[p_near_surface]
            max_t_near = max_t[p_near_surface]
            directions_near = direction[p_near_surface]
            
            # p_near_before = p_near.clone()
            # Keep stepping until delta > eps
            not_flipped = (t_near < max_t_near) & (delta_near < eps)
            while torch.any(not_flipped):
            
                # Take a step for the samples not flipped yet 
                p_near_not_flipped = p_near[not_flipped]
                delta_near_not_flipped = delta_near[not_flipped]
                t_near_not_flipped = t_near[not_flipped]
                
                # Take a step of size max( delta_near_not_flipped, eps/2 ) so that it doesn't get stuck with too small a step.
                p_near_not_flipped = p_near_not_flipped + torch.maximum(delta_near_not_flipped[:,None], torch.ones_like(delta_near_not_flipped)[:,None] * eps/2.0) * directions_near[not_flipped]
                t_near_not_flipped = t_near_not_flipped + torch.maximum(delta_near_not_flipped, torch.ones_like(delta_near_not_flipped) * eps/2.0)
        
                # Measure distance at new location
                with torch.no_grad():
                    delta_near_not_flipped = sdf_fn(p_near_not_flipped)
                sdf_cost += p_near_not_flipped.shape[0]
                delta_near_not_flipped = torch.abs(delta_near_not_flipped)
                
                # Copy back to these near surface samples 
                p_near[not_flipped] = p_near_not_flipped
                t_near[not_flipped] = t_near_not_flipped
                delta_near[not_flipped] = delta_near_not_flipped
                
                # Evaluate whether every sample is flipped 
                not_flipped = (t_near < max_t_near) & (delta_near < eps)
            
            # after flipping these samples to the other side of the surface, put those samples back with updated t and delta 
            p[p_near_surface] = p_near
            delta[p_near_surface] = delta_near
            t[p_near_surface] = t_near
        
        nc_far = not_converged
        p_far = p[nc_far,:]
        d_far = d[nc_far]
        delta_far = delta[nc_far]
        t_far = t[nc_far]

        # Take a step
        p_far = p_far + delta_far[:,None]/step_bound * direction[nc_far,:]
        t_far = t_far + delta_far/step_bound

        # Measure distance at new location
        with torch.no_grad():
            d_far = sdf_fn(p_far)
            
        sdf_cost += p_far.shape[0]
        delta_far = torch.abs(d_far)

        # Copy back to original tensors
        p[nc_far,:] = p_far
        d[nc_far] = d_far
        delta[nc_far] = delta_far
        t[nc_far] = t_far

        iter += 1
        not_converged = t < max_t
        
    if len(all_pts) > 0:
        all_pts = torch.stack(all_pts, dim=-2) # 5000, 480, 3
        all_pts_hit = torch.stack(all_pts_hit, dim=-1) # 5000, 480
        pts = all_pts.view(-1,3)[all_pts_hit.view(-1)]
        return pts, sdf_cost
    else: 
        return torch.zeros(0,3), sdf_cost
    
class ImplicitUniformSampler():
    
    def __init__(self, thresh=1e-4):    
        self.thresh = thresh
        
    def sample_rays(self, num_rays):
        
        dim = 3
        
        # sample line directions 
        dirs = np.random.randn(num_rays,dim)
        dirs = dirs / np.linalg.norm(dirs,axis=-1)[:,None] #[n,dim]
        dirs = dirs[:,None,:] #[n,1,3]
        
        # find the other normal and binormal basis 
        _,_,V = np.linalg.svd(dirs) 
        E = V[:,:,1:] #[n,3,2] 
        
        # sample random offsets in normal and binormal directions
        dirs = dirs[:,0,:] 
        U = np.sqrt(dim) * ( np.random.uniform(0,1,(num_rays,1,dim-1)) * 2.0 - 1.0 )  
        O = np.sum(U * E, axis=-1) 
        O = O + dirs * np.sqrt(dim)
        
        # determine if the ray O+D*t  intersects the [-1,1]^dim hypercube
        # using Slab method
        t_low = np.zeros((num_rays,dim))
        t_high = np.zeros((num_rays,dim))
        t_low = (-1.0 - O)/dirs 
        t_high = (1.0 - O)/dirs
        
        t_close = np.minimum(t_low, t_high)
        t_far = np.maximum(t_low,t_high)
        t_close = np.max(t_close, axis=-1)
        t_far = np.min(t_far,axis=-1)
        t = np.stack([t_close, t_far],axis=-1)
        keep = t_close < t_far
        
        dirs = dirs[keep]
        O = O[keep]
        t_close = t_close[keep]
        t_far = t_far[keep]
        O = O + dirs * t_close[:,None]
        T = t_far - t_close 
        n_current = O.shape[0]
        
        if (not np.any(keep)) and n_current > num_rays:
            return np.zeros((0,3)),np.zeros((0,3)),np.zeros((0))
        elif n_current == num_rays:
            return O, dirs, T
        elif n_current < num_rays: 
            O_current, D_current,T_current = self.sample_rays(num_rays-n_current)
            O = np.concatenate([O,O_current],axis=0)
            D = np.concatenate([dirs,D_current],axis=0)
            T = np.concatenate([T,T_current],axis=0)
            return O, D, T 
    
    def sample(self, sdf_func, num_rays): 
        lines_origins, lines_dirs, max_ts = self.sample_rays(num_rays)     
        # currently assume everything runs with pytorch on cuda 
        lines_origins = torch.from_numpy(lines_origins).cuda()
        lines_dirs = torch.from_numpy(lines_dirs).cuda()
        pts, _ = sphere_trace_modified(lines_origins, 
                                       lines_dirs, 
                                       sdf_fn=sdf_func, 
                                       max_t=torch.from_numpy(max_ts).cuda(), 
                                       eps=self.thresh) 
        return pts 

  
        