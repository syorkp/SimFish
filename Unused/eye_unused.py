

class EyeUnused:

    def read(self):
        grid = np.zeros((self.width, self.height))

        # For investigative purposes
        for i in range(self.photoreceptor_num):
            unique, counts = np.unique(full_set[i, :, :].get(), axis=0, return_counts=True)
            # counts = np.expand_dims(counts, 1)
            # frequencies = np.concatenate((unique, counts), axis=1)
            grid[unique[:, 0], unique[:, 1]] = counts*oversampling_ratio[i].get()
            grid = grid/self.n

        full_set = full_set.reshape(self.photoreceptor_num, -1, 2).astype(int)
        grid = np.zeros((self.width, self.height))
        grid[full_set[0, :, 0], full_set[0, :, 1]] = 1

        indexes = cp.zeros((self.width, self.height, 3), dtype=int)

        self.readings[:, 0] = (masked_arena_pixels[:, :, 0] * selected_points).sum(axis=1)
        self.readings[:, 1] = (masked_arena_pixels[:, :, 1] * selected_points).sum(axis=1)
        self.readings[:, 2] = (masked_arena_pixels[:, :, 2] * selected_points).sum(axis=1)

        total_sum = cp.zeros((self.photoreceptor_num, 3))

        # full_set = full_set.get()
        for i in range(self.photoreceptor_num):
            indexes[full_set[i, :, 0], full_set[i, :, 1], :] = 1
            total_sum[i] = (masked_arena_pixels * indexes).sum(axis=(0, 1))
            # indexes = np.unique(full_set[i, :, :], axis=0)
            # grid[indexes[:, 0], indexes[:, 1]] = 1
            indexes[:, :, :] = 0
            self.readings[i] = masked_arena_pixels[indexes[:, 0], indexes[:, 1]].sum(axis=0)
