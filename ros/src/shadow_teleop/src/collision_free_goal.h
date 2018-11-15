#pragma once

class Collision_freeGoal : public bio_ik::Goal {
    double mCollisionRadius = 0.03;

protected:
    double weight_;
    std::vector<std::string> mLinkNames_;
    std::vector<std::array<int, 2>> mCollisionPairs;

public:
    Collision_freeGoal(std::vector<std::string> mLinkNames, double weight = 1):mLinkNames_(mLinkNames), weight_(weight)
    {
      for(int i = 0;	i < mLinkNames_.size(); i = i+2)
        mCollisionPairs.push_back( {{i, i+1}} );
    }
    virtual void describe(bio_ik::GoalContext &context) const override {
      Goal::describe(context);
      context.setWeight(weight_);
      for (auto &linkName : mLinkNames_)
        context.addLink(linkName);
    }
    virtual double evaluate(const bio_ik::GoalContext &context) const override {
      double cost = 0.0;

      for (auto &p : mCollisionPairs) {
          double d = context.getLinkFrame(p[0]).getPosition().distance(
          context.getLinkFrame(p[1]).getPosition());
          // d is the bigger the better
          d = std::max(0.0, mCollisionRadius - d);
          cost += d * d * 10;
      }
      return cost;
    }
};
