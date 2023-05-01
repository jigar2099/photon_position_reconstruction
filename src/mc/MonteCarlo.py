import numpy as np
np.random.seed(42)
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class MonteCarlo:

    def __init__(self, input_uniq_shapes, photon_per_samp, required_examples, sample_size, samp_swap=True):
        self.input_uniq_shapes = input_uniq_shapes
        self.photon_per_samp = photon_per_samp
        self.required_examples = required_examples
        self.sample_size = sample_size
        self.shape_lengths = np.array([len(shape) for shape in input_uniq_shapes])
        self.samp_swap = samp_swap

    def remove_zeros(self, arr):
        return np.array(arr)[np.array(arr) != 0]

    def get_samp_from_ind(self, ind, shape_samples):
        # print(shape_samples)
        samples = [shape_samples[i] for i in ind]
        return samples

    def convert_samples_to_only_shapes(self, samples):
        shapes = [self.remove_zeros(i) for i in samples]
        return shapes

    def samplewise_shape_list_reshape(self, shape):
        reshaped_arr = np.array(shape).reshape(self.required_examples, self.photon_per_samp)
        return reshaped_arr

    def upper_lower_bounds(self, shape, peak_context=True):
        peak, _ = find_peaks(shape)
        shape_length = len(shape)
        if peak_context is False:
            if len(peak) == 0 and shape_length == 1:
                # print("Spike")
                lower_bound = 0
                upper_bound = self.sample_size - 1
            elif len(peak) == 0 and shape_length > 1:
                # print("Plateau")
                lower_bound = 0 - shape_length + 1
                upper_bound = 255 + shape_length - 1
            elif len(peak) != 0 and shape_length > 1:
                # print("Proper pulse")
                lower_bound = 0 - shape_length + 1
                upper_bound = 255 + shape_length - 1
            return lower_bound, upper_bound
        elif peak_context is True:
            if len(peak) == 0 and shape_length == 1:
                # print("Spike")
                lower_bound_pk = 0
                upper_bound_pk = self.sample_size - 1

                len_before_peak = 0
                len_after_peak = 0
                before_peak_part = 0
                after_peak_part = 0
            elif len(peak) == 0 and shape_length > 1:
                # print("Plateau")
                lower_bound_pk = 0 - shape_length + 1
                upper_bound_pk = self.sample_size - 1

                len_before_peak = 0
                len_after_peak = len(shape)
                before_peak_part = 0
                after_peak_part = shape
            elif len(peak) != 0 and shape_length > 1:
                # print("Proper pulse")
                before_peak_part = shape[:peak[0]]
                after_peak_part = shape[peak[0]:]

                len_before_peak = len(before_peak_part)
                len_after_peak = len(after_peak_part)

                lower_bound_pk = 0 - len_after_peak + 1
                upper_bound_pk = self.sample_size + len_before_peak - 1
            return lower_bound_pk, upper_bound_pk, len_before_peak, len_after_peak, before_peak_part, after_peak_part

    def get_ind_for_pulse_to_add(self, lower_bound, upper_bound):
        indice = np.random.randint(lower_bound, upper_bound, 1)
        return indice[0]

    def conditions_to_add_shapes(self, ind, shape, X, Y):
        if len(shape)==1:
            # single spike
            X[ind:ind+1]+=shape
            Y[ind]+=1
        else:
            peak_list, _ = find_peaks(shape)
            if not peak_list:
                full_plat_l_end = ind
                full_plat_h_end = ind + len(shape)
                if full_plat_l_end == 0:
                    X[:full_plat_h_end] += shape
                    Y[0] += 1
                elif full_plat_h_end == self.sample_size:
                    X[full_plat_l_end:] += shape
                    Y[full_plat_l_end] += 1
                elif full_plat_l_end>0 and full_plat_h_end<self.sample_size:
                    X[full_plat_l_end:full_plat_h_end] += shape
                    Y[full_plat_l_end] += 1
                elif full_plat_l_end<0 and full_plat_h_end>=0:
                    part_to_add = shape[abs(full_plat_l_end):]
                    X[:len(part_to_add)] += part_to_add
                    Y[0] += np.trapz(part_to_add)/np.trapz(shape)
                elif full_plat_l_end>0 and full_plat_h_end>self.sample_size:
                    part_to_add = shape[:abs(full_plat_l_end-self.sample_size)]
                    X[full_plat_l_end:] += part_to_add
                else:
                    print('----------Reconsider the MC simulation---------------------')
                    print(ind, shape)
                    print(full_plat_l_end, full_plat_h_end)
                    print('----------Reconsider the MC simulation---------------------')
                    
            else:
                # Pulse without plateau
                full_pulse_peak_ind = peak_list[0]
                full_pulse_l_end = ind-full_pulse_peak_ind
                full_pulse_h_end = full_pulse_l_end + len(shape)
                if ind==0: # peak at 0
                    part_to_add = shape[full_pulse_peak_ind:]
                    X[:len(part_to_add)] += part_to_add
                    Y[0] += np.trapz(part_to_add)/np.trapz(shape)
                elif ind==self.sample_size-1: # peak at end e.g, 255
                    part_to_add = shape[:full_pulse_peak_ind+1]
                    X[(self.sample_size-1)-full_pulse_peak_ind:] += part_to_add
                    Y[self.sample_size-1] += np.trapz(part_to_add)/np.trapz(shape)
                    
                elif ind<0: # peak beyond 0(negative)
                    part_to_add = shape[abs(full_pulse_l_end):]
                    X[:full_pulse_h_end] += part_to_add
                    Y[0] += np.trapz(part_to_add)/np.trapz(shape)
                elif ind>255: # peak beyond 255
                    part_to_add = shape[:(self.sample_size-full_pulse_l_end)]
                    X[full_pulse_l_end:] += part_to_add
                    fraction_of_pulse = np.trapz(part_to_add)/np.trapz(shape)
                    Y[255] += fraction_of_pulse
                elif ind>0:
                    if full_pulse_l_end>0 and full_pulse_h_end<self.sample_size:
                        X[full_pulse_l_end:full_pulse_h_end] += shape
                        Y[ind] += 1
                    elif full_pulse_l_end<0 and full_pulse_h_end<self.sample_size:
                        part_to_add = shape[abs(ind):]
                        X[:len(part_to_add)] += part_to_add
                        Y[ind] += np.trapz(part_to_add)/np.trapz(shape)
                    elif full_pulse_l_end>0 and full_pulse_h_end>self.sample_size:
                        part_to_add = shape[:abs(full_pulse_l_end-self.sample_size)]
                        X[full_pulse_l_end:] += part_to_add
                        Y[ind] += np.trapz(part_to_add)/np.trapz(shape)
                    elif full_pulse_l_end == 0 and full_pulse_h_end<self.sample_size:
                        X[:len(shape)] += shape
                        Y[ind] += 1
                    else: #full_pulse_l_end>0 and full_pulse_h_end==self.sample_size-1:
                        X[full_pulse_l_end:] += shape
                        Y[ind] += 1
                        
                
        
        
    def shapes_according_to_samples(self):
        samples_to_shapes = self.convert_samples_to_only_shapes(self.input_uniq_shapes)
        select_shape_ind_from_uniq_shapes = np.random.randint(len(self.input_uniq_shapes), size=(
        self.required_examples, self.photon_per_samp)).astype(int)
        select_shape_ind_from_uniq_shapes1 = np.random.randint(len(self.input_uniq_shapes), size=(
        self.required_examples, self.photon_per_samp)).astype(int)
        selected_shapes_to_be_added_in_samples = [
            samples_to_shapes[k1]
            for k in select_shape_ind_from_uniq_shapes
            for k1 in k
        ]
        selected_shapes_to_be_added_in_samples1 = [
            samples_to_shapes[k01]
            for k0 in select_shape_ind_from_uniq_shapes1
            for k01 in k0
        ]
        samplewise_reshaped_shapelist = self.samplewise_shape_list_reshape(selected_shapes_to_be_added_in_samples)
        samplewise_reshaped_shapelist1 = self.samplewise_shape_list_reshape(selected_shapes_to_be_added_in_samples1)
        return samplewise_reshaped_shapelist, samplewise_reshaped_shapelist1

    def correlation_manipulation_condition(self, j2,
                                           lb, ub, shape,
                                           lb1, ub1, shape1):
        get_ind_randomly_for_pulse_peak_to_add = self.get_ind_for_pulse_to_add(lb, ub)
        get_ind_randomly_for_pulse_peak_to_add1 = self.get_ind_for_pulse_to_add(lb1, ub1)
        
        if j2 == 1 and get_ind_randomly_for_pulse_peak_to_add<self.sample_size-1 and get_ind_randomly_for_pulse_peak_to_add>0:
            get_ind_randomly_for_pulse_peak_to_add1 = get_ind_randomly_for_pulse_peak_to_add
            if self.samp_swap is True:
                lb1 = lb;
                ub1 = ub;
                shape1 = shape
            else:
                pass
        elif j2 == 0:
            get_ind_randomly_for_pulse_peak_to_add = self.get_ind_for_pulse_to_add(lb, ub)
            get_ind_randomly_for_pulse_peak_to_add1 = self.get_ind_for_pulse_to_add(lb1, ub1)
        return get_ind_randomly_for_pulse_peak_to_add, get_ind_randomly_for_pulse_peak_to_add1, shape, shape1

    
    def MC2(self):
        arr_x = np.zeros((self.required_examples, self.sample_size))
        arr_y = np.zeros((self.required_examples, self.sample_size))
        arr_x1 = arr_x.copy()
        arr_y1 = arr_y.copy()
        
        ch1_ch2_corr_pattern = []
        ch1_pulse_peak_ind = []
        ch2_pulse_peak_ind = []
        
        samplewise_reshaped_shapelist, samplewise_reshaped_shapelist1 = self.shapes_according_to_samples()
        for i, X, Y, X1, Y1 in zip(range(self.required_examples), arr_x, arr_y, arr_x1, arr_y1):
            # This for loop gives an array of shapes which are selected to be added in a sample data
            corr_arr = np.random.choice([0, 1], size=self.photon_per_samp, replace=True)
            pulse_add_ind = []
            pulse_add_ind1= []
            for j, j1, j2 in zip(samplewise_reshaped_shapelist[i], samplewise_reshaped_shapelist1[i], corr_arr):
                # This loop is for accessing those shapes one by one
                # here j is the pulse
                lb, ub, len_bp, len_ap, bp_part, ap_part = self.upper_lower_bounds(j)
                lb1, ub1, len_bp1, len_ap1, bp_part1, ap_part1 = self.upper_lower_bounds(j1)
                get_ind_randomly_for_pulse_peak_to_add, get_ind_randomly_for_pulse_peak_to_add1, shape, shape1 = self.correlation_manipulation_condition( j2, lb, ub, j, lb1, ub1, j1)
                self.conditions_to_add_shapes(get_ind_randomly_for_pulse_peak_to_add, shape, X, Y)
                self.conditions_to_add_shapes(get_ind_randomly_for_pulse_peak_to_add1, shape1, X1, Y1)
                pulse_add_ind.append(get_ind_randomly_for_pulse_peak_to_add)
                pulse_add_ind1.append(get_ind_randomly_for_pulse_peak_to_add1)
            ch1_ch2_corr_pattern.append(corr_arr)
            ch1_pulse_peak_ind.append(pulse_add_ind)
            ch2_pulse_peak_ind.append(pulse_add_ind1)
            
        return arr_x, arr_y, arr_x1, arr_y1, ch1_ch2_corr_pattern, ch1_pulse_peak_ind, ch2_pulse_peak_ind


#input_uniq_shapes = np.load('./src/data/shaula_00000-003_ch0_shapes.npy')
#photon_per_samp = 15
#required_examples = 100
#sample_size = 256

#q = MonteCarlo(input_uniq_shapes, photon_per_samp, required_examples, sample_size, samp_swap=True)
#x, y, x1, y1, cor_pat, ch1_ind, ch2_ind = q.MC2()
