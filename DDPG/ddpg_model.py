import os, sys
import tensorflow as tf
from math import sqrt


class Model(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 optimizer=None,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 tau = 0.001,
                 sess=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau

        #tf.reset_default_graph()
        self.sess = sess or tf.Session()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        global_step_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
        self.sess.run(tf.variables_initializer(global_step_vars))

        self.actor_scope = "actor_net"
        with tf.name_scope(self.actor_scope):
            self.actor = DDPG_Actor(self.state_dim,
                        self.action_dim,
                        learning_rate=self.actor_learning_rate,
                        tau=self.tau,
                        scope=self.actor_scope,
                        sess=self.sess)

        self.critic_scope = "critic_net"
        with tf.name_scope(self.critic_scope):
            self.critic = DDPG_Critic(self.state_dim,
                        self.action_dim,
                        learning_rate=self.critic_learning_rate,
                        tau=self.tau,
                        scope=self.critic_scope,
                        sess=self.sess)

    def update(self, state_batch, action_batch, y_batch, sess=None):
        sess = sess or self.sess
        self.critic.update_source_critic_net(state_batch, action_batch, y_batch, sess)
        action_batch_for_grad = self.actor.predict_action_source_net(state_batch, sess)
        action_grad_batch = self.critic.get_action_grads(state_batch, action_batch_for_grad, sess)
        self.actor.update_source_actor_net(state_batch, action_grad_batch, sess)

        self.critic.update_target_critic_net(sess)
        self.actor.update_target_actor_net(sess)

    def predict_action(self, observation, sess=None):
        sess = sess or self.sess
        return self.actor.predict_action_source_net(observation, sess)

