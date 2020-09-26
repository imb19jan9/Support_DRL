import tkinter as tk
from tkinter import ttk
import numpy as np
import torch as th
from stable_baselines3 import PPO

from env import SupportEnv, LegalActionWrapper, ImageToPyTorch, ScaledFloatFrame


class App(tk.Tk):
    def __init__(self, env, model):
        super().__init__()

        self.env = env
        self.model = model

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, width=600, height=600)
        self.canvas.grid(row=0, column=0)

        self.control_panel = tk.Frame(self)
        self.next_button = ttk.Button(
            self.control_panel, text="Next", command=self.on_next
        )
        self.next_button.grid(row=0, column=0, sticky='ew')
        self.reset_button = ttk.Button(
            self.control_panel, text="Reset", command=self.on_reset
        )
        self.reset_button.grid(row=0, column=1, sticky='ew')
        self.show_pi_button = ttk.Button(
            self.control_panel, text="Show Pi", command=self.show_pi
        )
        self.show_pi_button.grid(row=0, column=2, sticky='ew')
        self.skip_button = ttk.Button(
            self.control_panel, text="Skip", command=self.skip
        )
        self.skip_button.grid(row=0, column=3, sticky='ew')
        self.control_panel.grid(row=1, column=0, sticky='esw')
        self.control_panel.columnconfigure((0,1,2,3), weight=1)

        self.after(200, self.setup)

    def setup(self):
        self.done = False
        self.obs = self.env.reset()
        cell_size = self.canvas.winfo_width() // self.env.width
        self.cells = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                upperleft = (j * cell_size, i * cell_size)
                lowerright = ((j + 1) * cell_size, (i + 1) * cell_size)
                if self.env.model[i, j] > 0:
                    id = self.canvas.create_rectangle(
                        upperleft, lowerright, fill="gray80"
                    )
                else:
                    id = self.canvas.create_rectangle(
                        upperleft, lowerright, fill="white"
                    )

                self.cells[f"({i},{j})"] = id

    def on_next(self):
        if self.done:
            self.on_reset()
            return

        row = self.env.action_row
        action = self.model.policy._predict(
            th.FloatTensor(self.obs).unsqueeze(0).to(th.device("cuda")),
            deterministic=True,
        ).item()

        self.obs, _, self.done, _ = self.env.step(action)

        id = self.cells[f"({row},{action})"]
        self.canvas.itemconfig(id, fill="gray20")

    def on_reset(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.setup()

    def show_pi(self):
        if self.done:
            return

        obs = th.FloatTensor(self.obs).unsqueeze(0).to(th.device("cuda"))

        row = self.env.action_row
        for i in range(self.env.action_space.n):
            if self.env.is_valid_action(i):
                action = th.LongTensor([i]).unsqueeze(0).to(th.device("cuda"))
                value, log_prob, entropy = self.model.policy.evaluate_actions(
                    obs, action
                )
                prob = th.exp(log_prob).item()
                if prob > 0.5:
                    rgb = 255, 0, int(255 * (-2 * prob + 2))
                else:
                    rgb = int(255 * (2 * prob)), 0, 255

                hex = "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])
                id = self.cells[f"({row},{i})"]
                self.canvas.itemconfig(id, fill=hex)

    def skip(self):
        while(not self.done):
            self.on_next()


if __name__ == "__main__":
    env = ScaledFloatFrame(
        ImageToPyTorch(LegalActionWrapper(SupportEnv(8, zoffset=3, reward=0.2, penalty=0.01)))
    )
    model = PPO.load("./rl_model")
    app = App(env, model)
    app.geometry('800x800')
    app.resizable(width=False, height=False)
    app.mainloop()