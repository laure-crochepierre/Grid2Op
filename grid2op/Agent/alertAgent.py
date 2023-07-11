# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Agent.baseAgent import BaseAgent
import pandas as pd


class AlertAgent(BaseAgent):
    """
    This is a :class:`AlertAgent` example, which will attempt to reconnect powerlines and send alerts on the worst possible attacks: for each disconnected powerline
    that can be reconnected, it will simulate the effect of reconnecting it. And reconnect the one that lead to the
    highest simulated reward. It will also simulate the effect of having a line disconnection on attackable lines and raise alerts for the worst ones

    """

    def __init__(self, action_space,percentage_alert=30,simu_step=0,controler=RecoPowerlineAgent):
        super().__init__(action_space=action_space)
        self.percentage_alert = percentage_alert
        self.simu_step=simu_step
        if isinstance(controler, type):
            # controler is passed as a class, and not as an object
            self.controler = controler(action_space)
        else:
            self.controler = controler()

        self.alertable_line_ids=self.action_space.alertable_line_ids

        #simu d'analyse de sécurité à chaque pas de temps sur les lignes attaquées
        self.n_alertable_lines=len(self.alertable_line_ids)

        self.N_1_actions=[self.action_space({"set_line_status": [(id_, -1)]}) for id_ in self.alertable_line_ids]

    
    def reset(self, obs): 
        
        self.nb_overloads=np.zeros(self.n_alertable_lines)
        self.rho_max_N_1=np.zeros(self.n_alertable_lines)

    def act(self, observation, reward, done=False):
        action=self.controler.act(observation, reward, done=done)
        
        #simu d'analyse de sécurité à chaque pas de temps sur les lignes attaquées

        # test which backend to know which method to call
        for i, action_to_simulate in enumerate(self.N_1_actions):

            #check that line is not already disconnected
            if (observation.line_status[self.alertable_line_ids[i]]):
                (
                    simul_obs,
                    simul_reward,
                    simul_has_error,
                    simul_info,
                ) = observation.simulate(action_to_simulate,time_step=self.simu_step)

                rho_simu=simul_obs.rho
                if(not simul_has_error):
                    self.nb_overloads[i]=np.sum(rho_simu >= 1)
                    self.rho_max_N_1[i]=np.max(rho_simu)

        indices = self.rho_max_N_1.argsort(axis=0)[::-1]

        #alerts to send
        indices_to_keep=list(indices[0:int(self.percentage_alert/100*self.n_alertable_lines)])
        action.raise_alert = [i for i in indices_to_keep]

        return action
