import torch
import torch.optim as optim

from .utils import _init_preconditioner, _update_preconditioner, _project, beta_debias, exp_avg_sq_


class ForeachSOAP(optim.Optimizer):
    """
    SFPaLMForeachSOAP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP

        ScheduleFree:
            The Road Less Scheduled
            Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
            https://arxiv.org/abs/2405.15682
            https://github.com/facebookresearch/schedule_free

        Beta2 Schedule:
            PaLM: Scaling Language Modeling with Pathways
            Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel
            https://arxiv.org/abs/2204.02311
    """

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1):
        defaults = {"lr": lr, "betas": betas, "shampoo_beta": shampoo_beta, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps}
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            vals = []
            step = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                merge_dims = group['merge_dims']
                max_precond_dim = group['max_precond_dim']
                precondition_1d = group['precondition_1d']

                grad = p.grad.float()
                p.grad = None

                state = self.state[p]
                step = state['step'] = state.get("step", -1) + 1

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    _init_preconditioner(grad, state, max_precond_dim, precondition_1d, merge_dims)
                    _update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = _project(grad, state['Q'], merge_dims, max_precond_dim, False)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                vals.append((p, grad, grad_projected, exp_avg, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, exp_avg, exp_avg_sq = zip(*vals)
            beta1, beta2 = group["betas"]

            old_debiased1 = beta_debias(beta1, step)
            old_debiased2 = beta_debias(beta2, step)

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            torch._foreach_mul_(exp_avg, old_debiased1)
            torch._foreach_add_(exp_avg, grad, alpha=1 - old_debiased1)
            denom = exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, group['eps'])

            for p, g, ea, d in zip(p_list, grad, exp_avg, denom):
                state = self.state[p]
                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = _project(ea, state['Q'], merge_dims, max_precond_dim, False)

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                # CANT DO /= HERE AS EXP_AVG MAY POINT TO THE BUFFER
                d.set_(_project(exp_avg_projected / d, state['Q'], merge_dims, max_precond_dim, True))

                _update_preconditioner(g, state, max_precond_dim, merge_dims, precondition_1d, 1 - old_debiased2,
                                       step > 0 and step % group['precondition_frequency'] == 0)

            # Why does this have to be rebiased here?
            step_size = -group["lr"] * min(step / group['warmup_steps'], 1)
            torch._foreach_add_(p_list, denom, alpha=step_size)  # projected back grad
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(p_list, p_list, alpha=step_size * group["weight_decay"])
        return loss