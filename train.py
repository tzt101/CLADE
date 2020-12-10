"""
Copyright (C) University of Science and Technology of China.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from data.base_dataset import repair_data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from options.test_options import TestOptions
import tqdm
from util import html
from util.util import tensor2im, tensor2label

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

if opt.train_eval:
    # val_opt = TestOptions().parse()
    original_flip = opt.no_flip
    opt.no_flip = True
    opt.phase = 'test'
    opt.isTrain = False
    dataloader_val = data.create_dataloader(opt)
    val_visualizer = Visualizer(opt)
    # # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    opt.phase = 'train'
    opt.isTrain = True
    opt.no_flip = original_flip
    # process for calculate FID scores
    from inception import InceptionV3
    from fid_score import *
    import pathlib
    # define the inceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.eval_dims]
    eval_model = InceptionV3([block_idx]).cuda()
    # load real images distributions on the training set
    mu_np_root = os.path.join('datasets/train_mu_si',opt.dataset_mode,'m.npy')
    st_np_root = os.path.join('datasets/train_mu_si',opt.dataset_mode,'s.npy')
    m0, s0 = np.load(mu_np_root), np.load(st_np_root)
    # load previous best FID
    if opt.continue_train:
        fid_record_dir = os.path.join(opt.checkpoints_dir, opt.name, 'fid.txt')
        FID_score, _ = np.loadtxt(fid_record_dir, delimiter=',', dtype=float)
    else:
        FID_score = 1000
else:
    FID_score = 1000      

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            if opt.train_eval:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter, FID_score)
            else:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        # if iter_counter.needs_displaying():
        #     visuals = OrderedDict([('input_label', data_i['label']),
        #                            ('synthesized_image', trainer.get_latest_generated()),
        #                            ('real_image', data_i['image'])])
        #     visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter(FID_score)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.eval_epoch_freq == 0 and opt.train_eval:
        # generate fake image
        trainer.pix2pix_model.eval()
        print('start evalidation .... ')
        if opt.use_vae:
            flag = True
            opt.use_vae = False
        else:
            flag = False
        for i, data_i in enumerate(dataloader_val):
            if data_i['label'].size()[0] != opt.batchSize:
                if opt.batchSize > 2*data_i['label'].size()[0]:
                    print('batch size is too large')
                    break
                data_i = repair_data(data_i, opt.batchSize)
            generated = trainer.pix2pix_model(data_i, mode='inference')
            img_path = data_i['path']
            for b in range(generated.shape[0]):
                tmp = tensor2im(generated[b])
                visuals = OrderedDict([('input_label', data_i['label'][b]),
                                    ('synthesized_image', generated[b])])
                val_visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        webpage.save()
        trainer.pix2pix_model.train()
        if flag:
            opt.use_vae = True
        # cal fid score
        fake_path = pathlib.Path(os.path.join(web_dir, 'images/synthesized_image/'))
        files = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))
        m1, s1 = calculate_activation_statistics(files, eval_model, 1, opt.eval_dims, True, images=None)
        fid_value = calculate_frechet_distance(m0, s0, m1, s1)
        visualizer.print_eval_fids(epoch, fid_value, FID_score)
        # save the best model if necessary
        if fid_value < FID_score:
            FID_score = fid_value
            trainer.save('best')

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
