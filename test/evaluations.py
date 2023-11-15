import torch


def missing_variable_testing(test_loader, augmenter, normalizer):
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            if params.grid_type == "non uniform":
                '''
                Assume non uniform grids requires
                updating grid for every sample. We need to
                suppy the grid.

                last 3 channel is displacement, taking (x,y), z is 0
                '''
                with torch.no_grad():
                    if stage == 'ssl':
                        out_grid_displacement = x[0, :, -3:-1].clone().detach()
                        in_grid_displacement = x[0, :, -3:-1].clone().detach()
                    else:
                        out_grid_displacement = y[0, :, -3:-1].clone().detach()
                        in_grid_displacement = x[0, :, -3:-1].clone().detach()
            else:
                out_grid_displacement = None
                in_grid_displacement = None

            if normalizer is not None:
                with torch.no_grad():
                    x, y = normalizer(x), normalizer(y)

            batch_size = x.shape[0]
            out = model(x, out_grid_displacement, in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]

            ntest += 1
            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            test_l2 += loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                              ).item() / torch.norm(target.reshape(batch_size, -1), p=2, dim=-1).item()

    test_l2 /= ntest
    t2 = default_timer()

    wandb.log({'test_error_' + stage: test_l2}, commit=True)
    print(f"Test Error  {stage}: ", test_l2)
