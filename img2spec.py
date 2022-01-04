import struct
import wave

import matplotlib.pyplot as plt
import numpy
import scipy.ndimage
import scipy.signal
from PIL import Image


def show_img(arr):
    pil_image = Image.fromarray(arr)
    pil_image.show()


def load_image(img_path, freq_step, samp_rate, duration):
    with Image.open(img_path) as img:
        img = img.convert("L")
        img_data = numpy.array(img)
        max_sample = duration * samp_rate
        zoom_factor = ((samp_rate / 2) / freq_step / img_data.shape[0], max_sample / img_data.shape[1])
        img_data = scipy.ndimage.zoom(img_data, zoom_factor, order=0)
        return img_data


def rescale(data, upper, lower):
    return [
        lower + (element - data.min()) * (upper - lower) / (data.max() - data.min())
        for
        element
        in
        data
    ]


def rescale_val(val, scaled_max, scaled_min, max_val, min_val):
    return scaled_min + (val - min_val) * (scaled_max - scaled_min) / (max_val - min_val)


def show_spectrogram(data, rate):
    # scipy.signal.spectrogram(numpy.array(normalized), sampling_rate)
    freqs, times, spectrogram = scipy.signal.spectrogram(numpy.array(data), rate)
    plt.pcolormesh(times, freqs, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def write_to_wav(out_file, data, rate, lpad=0, rpad=0):
    zero_byte = struct.pack("h", 0)

    with wave.open(out_file, "w") as wav_file:
        wav_file.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))

        for _ in range(int(lpad)):
            wav_file.writeframes(zero_byte)

        for i, sample in enumerate(data):
            to_write = struct.pack("h", int(sample * 32767.0))
            wav_file.writeframes(to_write)

        for _ in range(int(rpad)):
            wav_file.writeframes(zero_byte)


def gen_wave(freq, amp, samp_step, samp_rate):
    # amount of periods fitting in one samp_step
    periods = (samp_step * freq) / samp_rate
    samples = numpy.arange(samp_step)
    return amp * numpy.sin(periods * 2 * numpy.pi * samples / samp_step)


def img2spec(
        img_path,
        out_path,
        duration,
        img_freq_interval,  # tuple
        img_time_interval,  # tuple
        samp_rate,
        freq_step=175,
        samp_step=1050,
):
    img_duration = img_time_interval[1] - img_time_interval[0]
    img_data = load_image(img_path, freq_step=freq_step, samp_rate=samp_rate,
                          duration=img_duration)
    img_data = numpy.flip(img_data, axis=0)
    img_data = numpy.transpose(img_data)

    out_signal = numpy.array([])

    for x in range(0, img_data.shape[0], samp_step):
        signal = numpy.zeros(samp_step)
        # value 255 is white
        if img_data[x].min() == 255:
            out_signal = numpy.append(out_signal, signal)
            continue
        for y in range(img_data.shape[1] - 1, 0, -1):
            if img_data[x][y] < 255:
                amp = 1 - img_data[x][y] / 255
                rescaled_freq = rescale_val(freq_step * y, img_freq_interval[1], img_freq_interval[0],
                                       freq_step * (img_data.shape[1] - 1), 0)
                signal += gen_wave(rescaled_freq, amp, samp_step, samp_rate)
        out_signal = numpy.append(out_signal, signal)

    rescaled_out_signal = rescale(out_signal, 1, -1)

    lpad = img_time_interval[0] * samp_rate
    rpad = (duration - img_time_interval[1]) * samp_rate

    write_to_wav(out_path, rescaled_out_signal, samp_rate, lpad=lpad, rpad=rpad)


img2spec(img_path="smiley_double.png", out_path="img2spec.wav", samp_rate=48000, duration=8.0,
         img_freq_interval=(8000, 10000), img_time_interval=(3.0, 5.0))
