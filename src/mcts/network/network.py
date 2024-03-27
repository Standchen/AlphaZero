import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = [
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=dim),
        ]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class TictactoeDualNetwork(nn.Module):
    def __init__(self, res_dim=128, num_res_blk=9):
        super().__init__()
        input_dim = 3
        board_area = 3**2
        out_dim = board_area

        p_channel = 4
        v_channel = 2

        # Root
        self.root = [
            nn.Conv2d(input_dim, res_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=res_dim),
            nn.ReLU(inplace=True),
        ]
        self.root += [ResidualBlock(dim=res_dim) for _ in range(num_res_blk)]

        # Policy head
        self.p_head = [
            nn.Conv2d(res_dim, p_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=p_channel),
            nn.ReLU(),
        ]
        self.p_head += [
            nn.Flatten(),
            nn.Linear(p_channel * board_area, out_dim),
            nn.Softmax(dim=1),
        ]

        # Value head
        self.v_head = [
            nn.Conv2d(res_dim, v_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=v_channel),
            nn.ReLU(),
        ]
        self.v_head += [
            nn.Flatten(),
            nn.Linear(v_channel * board_area, 1),
            nn.Tanh(),
        ]

        # Finalize.
        self.root = nn.Sequential(*self.root)
        self.p_head = nn.Sequential(*self.p_head)
        self.v_head = nn.Sequential(*self.v_head)

    def forward(self, x):
        r = self.root(x)
        p, v = self.p_head(r), self.v_head(r)
        return p, v


#############################################################################################


"""
class OthelloDualNetwork(nn.Module):
    def __init__(self, size=4, res_dim=128, num_res_blk=16):
        super().__init__()
        input_dim = 3
        board_area = size**2
        out_dim = board_area + 1

        p_channel = 4
        v_channel = 2

        # Root
        self.root = [
            nn.Conv2d(input_dim, res_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=res_dim),
            nn.ReLU(inplace=True),
        ]
        self.root += [ResidualBlock(dim=res_dim) for _ in range(num_res_blk)]

        # Policy head
        self.p_head = [
            nn.Conv2d(res_dim, p_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=p_channel),
            nn.ReLU(),
        ]
        self.p_head += [
            nn.Flatten(),
            nn.Linear(p_channel * board_area, out_dim),
            nn.Softmax(dim=1),
        ]

        # Value head
        self.v_head = [
            nn.Conv2d(res_dim, v_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=v_channel),
            nn.ReLU(),
        ]
        self.v_head += [
            nn.Flatten(),
            nn.Linear(v_channel * board_area, 1),
            nn.Tanh(),
        ]

        # Finalize.
        self.root = nn.Sequential(*self.root)
        self.p_head = nn.Sequential(*self.p_head)
        self.v_head = nn.Sequential(*self.v_head)

    def forward(self, x):
        r = self.root(x)
        p, v = self.p_head(r), self.v_head(r)
        return p, v
"""

# ------------------------------------------------------------

"""
class OthelloDualNetwork(nn.Module):
    def __init__(self, size=6, res_dim=128, num_res_blk=16):
        super().__init__()
        input_dim = 3
        board_area = size**2
        out_dim = board_area + 1

        p_channel = 4
        v_channel = 2

        # Root
        self.root = [
            nn.Conv2d(input_dim, res_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=res_dim),
            nn.ReLU(inplace=True),
        ]
        self.root += [ResidualBlock(dim=res_dim) for _ in range(num_res_blk)]

        # Policy head
        self.p_head = [
            nn.Conv2d(res_dim, p_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=p_channel),
            nn.ReLU(),
        ]
        self.p_head += [
            nn.Flatten(),
            nn.Linear(p_channel * board_area, out_dim),
            nn.Softmax(dim=1),
        ]

        # Value head
        self.v_head = [
            nn.Conv2d(res_dim, v_channel, kernel_size=1),
            nn.BatchNorm2d(num_features=v_channel),
            nn.ReLU(),
        ]
        self.v_head += [
            nn.Flatten(),
            nn.Linear(v_channel * board_area, 1),
            nn.Tanh(),
        ]

        # Finalize.
        self.root = nn.Sequential(*self.root)
        self.p_head = nn.Sequential(*self.p_head)
        self.v_head = nn.Sequential(*self.v_head)

    def forward(self, x):
        r = self.root(x)
        p, v = self.p_head(r), self.v_head(r)
        return p, v
"""

# ------------------------------------------------------------

class OthelloDualNetwork(nn.Module):
    def __init__(self, size=6, res_dim=512, num_res_blk=2):
        super().__init__()
        input_dim = 3
        board_area = size**2
        out_dim = board_area + 1

        # Root
        self.root = [
            nn.Conv2d(input_dim, res_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=res_dim),
            nn.ReLU(inplace=True),
        ]
        self.root += [ResidualBlock(dim=res_dim) for _ in range(num_res_blk)]

        self.root += [
            nn.Conv2d(res_dim, 128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        ]

        self.root += [
            nn.Flatten(),
            nn.Linear(128 * board_area, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
        ]

        # Policy head
        self.p_head = [
            nn.Linear(512, out_dim),
            nn.Softmax(dim=1),
        ]

        # Value head
        self.v_head = [
            nn.Linear(512, 1),
            nn.Tanh(),
        ]

        # Finalize.
        self.root = nn.Sequential(*self.root)
        self.p_head = nn.Sequential(*self.p_head)
        self.v_head = nn.Sequential(*self.v_head)

    def forward(self, x):
        r = self.root(x)
        p, v = self.p_head(r), self.v_head(r)
        return p, v

# ------------------------------------------------------------

"""
class OthelloDualNetwork(nn.Module):
    def __init__(self, size=6, res_dim=512, num_res_blk=2):
        super().__init__()
        # game params
        self.board_x, self.board_y = size, size
        self.action_size = (size**2) + 1

        self.num_channels = 512

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.softmax(pi, dim=1), torch.tanh(v)
"""


#############################################################################################


class PentagoDualNetwork(nn.Module):
    def __init__(self, res_dim=128, num_res_blk=16):
        super().__init__()
        input_dim = 3
        board_area = 6**2
        # out_dim = 8 * board_area

        p_channel_1 = 32
        p_channel_2 = 16
        v_channel = 2

        assert p_channel_2 > 8

        # Root
        self.root = []
        self.root += [nn.Conv2d(input_dim, res_dim, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(num_features=res_dim),
                      nn.ReLU(inplace=True)]
        self.root += [ResidualBlock(dim=res_dim) for _ in range(num_res_blk)]

        # Policy head
        self.p_head = []
        self.p_head += [nn.Conv2d(res_dim, p_channel_1, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_features=p_channel_1),
                        nn.ReLU()]
        self.p_head += [nn.Conv2d(p_channel_1, p_channel_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_features=p_channel_2),
                        nn.ReLU()]
        self.p_head += [nn.Conv2d(p_channel_2, 8, kernel_size=1),
                        nn.BatchNorm2d(num_features=8),
                        nn.ReLU()]
        self.p_head += [nn.Flatten(),
                        nn.Softmax(dim=1)]

        # Value head
        self.v_head = []
        self.v_head += [nn.Conv2d(res_dim, v_channel, kernel_size=1),
                        nn.BatchNorm2d(num_features=v_channel),
                        nn.ReLU()]
        self.v_head += [nn.Flatten(),
                        nn.Linear(v_channel * board_area, 1),
                        nn.Tanh()]

        # Finalize.
        self.root = nn.Sequential(*self.root)
        self.p_head = nn.Sequential(*self.p_head)
        self.v_head = nn.Sequential(*self.v_head)

    def forward(self, x):
        r = self.root(x)
        p, v = self.p_head(r), self.v_head(r)
        return p, v
