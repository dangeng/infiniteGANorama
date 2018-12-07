'''
Script to train the ranker
Should add some sort of image pool someday...?
'''

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

import pdb
import torch
from collections import OrderedDict

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    chkpt_D = torch.load('checkpoints/streetview_throttled/15_net_D.pth')
    chkpt_G = torch.load('checkpoints/streetview_throttled/15_net_G.pth')

    new_chkpt_D = OrderedDict()
    new_chkpt_G = OrderedDict()
    for k, v in chkpt_D.items():
        name = 'module.' + k # add `module`
        new_chkpt_D[name] = v
    for k, v in chkpt_G.items():
        name = 'module.' + k # add `module`
        new_chkpt_G[name] = v

    model.netD.load_state_dict(new_chkpt_D)
    model.netG.load_state_dict(new_chkpt_G)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            # optimize only discriminator
            model.forward()
            model.set_requires_grad(model.netD, True)
            model.optimizer_D.zero_grad()
            model.backward_D()
            model.optimizer_D.step()
            model.set_requires_grad(model.netD, False)

            # need this to prevent logger from complaining
            # because it wants to log the G loss, even though
            # we aren't updating it
            model.backward_G()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
