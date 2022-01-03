import struct

from PIL import Image
import numpy
import scipy.ndimage
import scipy.signal
import wave
import matplotlib.pyplot as plt


def load_image(img_path, freq_step, samp_rate, duration):
    with Image.open(img_path) as img:
        # convert to grayscale
        img = img.convert("L")
        img_data = numpy.array(img)
        # flip along the horizontal axis
        img_data = numpy.flip(img_data, axis=0)
        max_sample = duration * samp_rate
        # determine zoom factor
        zoom_factor = ((samp_rate / 2) / freq_step / img_data.shape[0], max_sample / img_data.shape[1])
        # apply zoom on image array
        img_data = scipy.ndimage.zoom(img_data, zoom_factor, order=0)
        return img_data


def max_min_norm(data, upper, lower):
    return [
        lower + (element - data.min()) * (upper - lower) / (data.max() - data.min())
        for
        element
        in
        data
    ]


def show_spectrogram(data, rate):
    # scipy.signal.spectrogram(numpy.array(normalized), sampling_rate)
    freqs, times, spectrogram = scipy.signal.spectrogram(numpy.array(data), rate)
    plt.pcolormesh(times, freqs, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def write_to_wav(out_file, data, rate):
    with wave.open(out_file, "w") as wav_file:
        wav_file.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        for i, sample in enumerate(data):
            print(f"writing sample {i + 1}/{len(data)}")
            to_write = struct.pack("h", int(sample * 32767.0))
            wav_file.writeframes(to_write)


def gen_wave(freq, amp, samp_step, samp_rate):
    # amount of periods fitting in one samp_step
    periods = (samp_step * freq) / samp_rate
    samples = numpy.arange(samp_step)
    return amp * numpy.sin(periods * 2 * numpy.pi * samples / samp_step)


def img2spec(img_path, out_path="img2spec.wav", freq_step=175, samp_step=1050, samp_rate=22050, duration=5.0):
    img_data = load_image(img_path, freq_step=freq_step, samp_rate=samp_rate, duration=duration)
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
                signal += gen_wave(y * freq_step, amp, samp_step, samp_rate)
        out_signal = numpy.append(out_signal, signal)

    normalized = max_min_norm(out_signal, 1, -1)
    # show_spectrogram(normalized, samp_rate)
    # show_spectrogram(out_signal, samp_rate)
    write_to_wav(out_path, normalized, samp_rate)


img2spec(img_path="gradient2.png", out_path="img2spec.wav", freq_step=25, samp_step=1050, samp_rate=22050)
